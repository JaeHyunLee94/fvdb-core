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

#include <nanovdb/NanoVDB.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>

#include <cstdint>

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

__global__ void
stencilConvKernel(const float *__restrict__ inputFeatures,
                  const float *__restrict__ weights,
                  const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *sourceGrid,
                  const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *targetGrid,
                  float *__restrict__ outputFeatures) {
    __shared__ float haloValues[kHaloSize][kHaloSize][kHaloSize];
    //

    const int leafID = blockIdx.x;
    const int tid   = threadIdx.x;

    const auto &outLeaf    = targetGrid->tree().template getFirstNode<0>()[leafID];
    const auto  leafOrigin = outLeaf.origin();
    const int   Lx         = leafOrigin[0];
    const int   Ly         = leafOrigin[1];
    const int   Lz         = leafOrigin[2];

    // ------------------------------------------------------------------
    // Phase 1: cooperative halo population.
    // 512 threads fill 1000 slots in two passes.  Pass 1 covers slots 0-511;
    // pass 2 covers slots 512-999 and only the first 488 threads participate.
    // ------------------------------------------------------------------
    auto srcAcc = sourceGrid->getAccessor();

    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int s = tid + pass * kThreads;
        if (s < kHaloVoxels) {
            const int          i = s / (kHaloSize * kHaloSize);
            const int          j = (s / kHaloSize) % kHaloSize;
            const int          k = s % kHaloSize;
            const nanovdb::Coord ijk(Lx + i - 1, Ly + j - 1, Lz + k - 1);
            const uint64_t     raw = srcAcc.getValue(ijk);
            haloValues[i][j][k]    = raw ? inputFeatures[raw - 1] : 0.0f;
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 2: per-thread output voxel accumulation.
    // ------------------------------------------------------------------
    // Changing this to bit shifting?
    const int li = (tid >> 6) & 0x7;
    const int lj = (tid >> 3) & 0x7;
    const int lk = tid & 0x7;

    const nanovdb::Coord outIJK(Lx + li, Ly + lj, Lz + lk);
    auto                 tgtAcc = targetGrid->getAccessor();
    const int64_t        outIdx = static_cast<int64_t>(tgtAcc.getValue(outIJK)) - 1;
    if (outIdx < 0) {
        return;
    }

    float sum = 0.0f;
    #pragma unroll
    for (int di = -1; di <= 1; ++di) {
        #pragma unroll
        for (int dj = -1; dj <= 1; ++dj) {
            #pragma unroll
            for (int dk = -1; dk <= 1; ++dk) {
                const int woff = (di + 1) * 9 + (dj + 1) * 3 + (dk + 1);
                sum += weights[woff] * haloValues[li + di + 1][lj + dj + 1][lk + dk + 1];
            }
        }
    }

    outputFeatures[outIdx] = sum;
}

} // namespace

torch::Tensor
stencilSparseConv(const torch::Tensor &inputFeatures,
                  const torch::Tensor &weights,
                  const GridBatchImpl &sourceGrid,
                  const GridBatchImpl &targetGrid) {
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

    stencilConvKernel<<<numTargetLeaves, kThreads, 0, stream>>>(
        inputFeatures.data_ptr<float>(),
        weights.data_ptr<float>(),
        sourceNanoGrid,
        targetNanoGrid,
        output.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

} // namespace ops
} // namespace detail
} // namespace fvdb
