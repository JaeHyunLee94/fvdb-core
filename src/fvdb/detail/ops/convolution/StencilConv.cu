// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// StencilConv.cu -- CTA-per-leaf scalar stencil sparse convolution.
//
// Each CTA is assigned one leaf node of the target grid.  The 10x10x10 input
// halo covering the 8x8x8 leaf region is cooperatively staged into shared
// memory via NanoVDB accessor lookups, then each thread in the CTA performs
// a 27-tap multiply-accumulate for a single active output voxel.

#include <fvdb/detail/ops/convolution/StencilConv.h>
#include <fvdb/detail/ops/convolution/StencilDescriptor.h>

#include <nanovdb/NanoVDB.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>

#include <cstdint>
#include <tuple>
#include <utility>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// Leaf is 8x8x8 = 512 voxels; with R=1 halo the footprint is 10x10x10 = 1000 voxels.
constexpr int kLeafSize   = 8;
constexpr int kHaloSize   = kLeafSize + 2;
constexpr int kHaloVoxels = kHaloSize * kHaloSize * kHaloSize;
constexpr int kLeafVoxels = kLeafSize * kLeafSize * kLeafSize;
constexpr int kThreads    = kLeafVoxels;

// Fold helper: compile-time unroll over StencilT::Taps.
template <typename Taps, std::size_t... Is>
__device__ __forceinline__ float
sumTapsImpl(const float *__restrict__ weights,
            const float (&halo)[kHaloSize][kHaloSize][kHaloSize],
            int li,
            int lj,
            int lk,
            std::index_sequence<Is...>) {
    float sum = 0.0f;
    ((sum +=
      weights[(std::tuple_element_t<Is, Taps>::di + 1) * 9 +
              (std::tuple_element_t<Is, Taps>::dj + 1) * 3 +
              (std::tuple_element_t<Is, Taps>::dk + 1)] *
      halo[li + std::tuple_element_t<Is, Taps>::di + 1]
          [lj + std::tuple_element_t<Is, Taps>::dj + 1]
          [lk + std::tuple_element_t<Is, Taps>::dk + 1]),
     ...);
    return sum;
}

template <typename StencilT>
__device__ __forceinline__ float
sumTaps(const float *__restrict__ weights,
        const float (&halo)[kHaloSize][kHaloSize][kHaloSize],
        int li,
        int lj,
        int lk) {
    using Taps = typename StencilT::Taps;
    return sumTapsImpl<Taps>(
        weights, halo, li, lj, lk,
        std::make_index_sequence<std::tuple_size_v<Taps>>{});
}

template <typename StencilT>
__global__ void
stencilConvKernel(const float *__restrict__ inputFeatures,
                  const float *__restrict__ weights,
                  const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *sourceGrid,
                  const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *targetGrid,
                  float *__restrict__ outputFeatures) {
    using SrcLeafT = nanovdb::NanoLeaf<nanovdb::ValueOnIndex>;

    __shared__ float          haloValues[kHaloSize][kHaloSize][kHaloSize];
    // 3x3x3 cache of source-leaf pointers covering the halo neighborhood.
    // Indexed as haloLeaves[leafI][leafJ][leafK], where each axis ∈ {0,1,2}
    // maps to the neighbor leaf offset {-1, 0, +1} along that axis. A null
    // pointer marks a leaf that does not exist in the source grid.
    __shared__ const SrcLeafT *haloLeaves[3][3][3];

    const int leafID = blockIdx.x;
    const int tid    = threadIdx.x;

    const auto &outLeaf    = targetGrid->tree().template getFirstNode<0>()[leafID];
    const auto  leafOrigin = outLeaf.origin();
    const int   Lx         = leafOrigin[0];
    const int   Ly         = leafOrigin[1];
    const int   Lz         = leafOrigin[2];

    // ------------------------------------------------------------------
    // Phase 0: 27-leaf neighborhood lookup (one tree walk per leaf, not per
    // halo slot). First 27 threads each probe one leaf of the 3x3x3 block
    // around the current output leaf; the remaining threads idle. Reduces
    // tree traversals from ~1000/CTA to 27/CTA.
    // ------------------------------------------------------------------
    const auto &srcTree = sourceGrid->tree();
    if (tid < 27) {
        const int li = tid / 9;        // 0, 1, 2
        const int lj = (tid / 3) % 3;
        const int lk = tid % 3;
        const nanovdb::Coord leafOri(Lx + (li - 1) * kLeafSize,
                                     Ly + (lj - 1) * kLeafSize,
                                     Lz + (lk - 1) * kLeafSize);
        haloLeaves[li][lj][lk] = srcTree.root().probeLeaf(leafOri);
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 1: cooperative halo population using cached leaf pointers.
    // 512 threads fill 1000 slots in two passes (pass 2 uses 488 threads).
    // Each slot does one O(1) leaf::getValue(localOffset) — no tree walk.
    // ------------------------------------------------------------------
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int s = tid + pass * kThreads;
        if (s < kHaloVoxels) {
            const int i = s / (kHaloSize * kHaloSize);
            const int j = (s / kHaloSize) % kHaloSize;
            const int k = s % kHaloSize;

            // Global coord of this halo slot.
            const int gx = Lx - 1 + i;
            const int gy = Ly - 1 + j;
            const int gz = Lz - 1 + k;

            // Which of the 27 neighbor leaves owns (gx,gy,gz)?
            //   i=0       -> leafI = 0   (leaf at Lx-8)
            //   i=1..8    -> leafI = 1   (current leaf at Lx)
            //   i=9       -> leafI = 2   (leaf at Lx+8)
            // (gx >> 3) - (Lx >> 3) computes the signed leaf-index delta.
            const int leafI = (gx >> 3) - (Lx >> 3) + 1;
            const int leafJ = (gy >> 3) - (Ly >> 3) + 1;
            const int leafK = (gz >> 3) - (Lz >> 3) + 1;

            const SrcLeafT *leaf = haloLeaves[leafI][leafJ][leafK];
            float           val  = 0.0f;
            if (leaf != nullptr) {
                // Local offset within the leaf: (x & 7) << 6 | (y & 7) << 3 | (z & 7).
                // Matches NanoLeaf<>::CoordToOffset for LOG2DIM=3.
                const uint32_t localOff = ((gx & 0x7) << 6) | ((gy & 0x7) << 3) | (gz & 0x7);
                const uint64_t raw      = leaf->getValue(localOff);
                if (raw) {
                    val = inputFeatures[raw - 1];
                }
            }
            haloValues[i][j][k] = val;
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 2: per-thread output voxel accumulation.
    // ------------------------------------------------------------------
    const int li = (tid >> 6) & 0x7;
    const int lj = (tid >> 3) & 0x7;
    const int lk = tid & 0x7;

    // outLeaf already covers (Lx+li, Ly+lj, Lz+lk) by construction, so skip
    // the root-to-leaf tree walk; compute the packed local offset directly.
    const uint32_t outLocalOff = (li << 6) | (lj << 3) | lk;
    const int64_t  outIdx      = static_cast<int64_t>(outLeaf.getValue(outLocalOff)) - 1;
    if (outIdx < 0) {
        return;
    }

    const float sum = sumTaps<StencilT>(weights, haloValues, li, lj, lk);

    outputFeatures[outIdx] = sum;
}

} // namespace

torch::Tensor
stencilSparseConv(const torch::Tensor &inputFeatures,
                  const torch::Tensor &weights,
                  const GridBatchImpl &sourceGrid,
                  const GridBatchImpl &targetGrid,
                  StencilKind          kind) {
    TORCH_CHECK(inputFeatures.is_cuda(), "inputFeatures must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(inputFeatures.scalar_type() == torch::kFloat32,
                "inputFeatures must be float32");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(inputFeatures.is_contiguous(), "inputFeatures must be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");

    TORCH_CHECK(inputFeatures.dim() == 1 || inputFeatures.dim() == 2,
                "inputFeatures must be 1-D or 2-D, got ",
                inputFeatures.dim(),
                "-D");
    if (inputFeatures.dim() == 2) {
        TORCH_CHECK(inputFeatures.size(1) == 1,
                    "StencilConv requires in_channels=1, got ",
                    inputFeatures.size(1));
    }

    TORCH_CHECK(weights.dim() == 5, "weights must be 5-D [1,1,3,3,3]");
    TORCH_CHECK(weights.size(0) == 1 && weights.size(1) == 1 && weights.size(2) == 3 &&
                    weights.size(3) == 3 && weights.size(4) == 3,
                "StencilConv requires weights shape [1,1,3,3,3], got [",
                weights.size(0), ",", weights.size(1), ",", weights.size(2), ",",
                weights.size(3), ",", weights.size(4), "]");

    TORCH_CHECK(sourceGrid.device() == targetGrid.device(),
                "sourceGrid and targetGrid must be on the same device");
    TORCH_CHECK(sourceGrid.device() == inputFeatures.device(),
                "inputFeatures and grids must be on the same device");
    TORCH_CHECK(sourceGrid.batchSize() == 1,
                "StencilConv currently supports batch size 1, got ",
                sourceGrid.batchSize());
    TORCH_CHECK(targetGrid.batchSize() == 1,
                "StencilConv currently supports batch size 1, got ",
                targetGrid.batchSize());

    const int64_t N_in  = sourceGrid.totalVoxels();
    const int64_t N_out = targetGrid.totalVoxels();
    TORCH_CHECK(inputFeatures.size(0) == N_in,
                "inputFeatures first dim (",
                inputFeatures.size(0),
                ") must match sourceGrid totalVoxels (",
                N_in,
                ")");

    auto opts   = inputFeatures.options();
    auto output = torch::zeros({N_out, 1}, opts);

    const uint32_t numTargetLeaves = targetGrid.numLeavesAt(0);
    if (numTargetLeaves == 0 || N_out == 0) {
        return output;
    }

    const auto *sourceNanoGrid =
        sourceGrid.nanoGridHandle().template deviceGrid<nanovdb::ValueOnIndex>();
    const auto *targetNanoGrid =
        targetGrid.nanoGridHandle().template deviceGrid<nanovdb::ValueOnIndex>();
    TORCH_CHECK(sourceNanoGrid != nullptr, "Failed to get device sourceGrid");
    TORCH_CHECK(targetNanoGrid != nullptr, "Failed to get device targetGrid");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (kind) {
    case StencilKind::Dense27:
        stencilConvKernel<Dense27Stencil>
            <<<numTargetLeaves, kThreads, 0, stream>>>(
                inputFeatures.data_ptr<float>(),
                weights.data_ptr<float>(),
                sourceNanoGrid,
                targetNanoGrid,
                output.data_ptr<float>());
        break;
    case StencilKind::Laplace7:
        stencilConvKernel<Laplace3DStencil>
            <<<numTargetLeaves, kThreads, 0, stream>>>(
                inputFeatures.data_ptr<float>(),
                weights.data_ptr<float>(),
                sourceNanoGrid,
                targetNanoGrid,
                output.data_ptr<float>());
        break;
    default:
        TORCH_CHECK(false, "StencilConv: unknown StencilKind ", static_cast<int>(kind));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

} // namespace ops
} // namespace detail
} // namespace fvdb
