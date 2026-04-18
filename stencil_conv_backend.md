# Stencil Convolution Backend Design

## 1. Motivation and Goals

fVDB's existing convolution backends — `GatherScatterDefault` and `PredGatherIGemm` — both require a **kernel map (kmap)**: a precomputed data structure that explicitly enumerates all active `(input_voxel, output_voxel)` pairs for every kernel offset. This precomputation is a meaningful up-front cost, and the resulting kmap is a substantial data structure that must be read back during execution.

For **scalar** convolutions (`in_channels = out_channels = 1`) with a **small fixed-radius stencil** (R=1, i.e. all taps fit in a 3×3×3 box), a different execution strategy is possible and potentially more efficient: launch one CUDA CTA per output leaf node, stage the entire 10×10×10 input halo into shared memory, and perform all 27-tap multiply-accumulates from smem. This avoids building a kmap entirely, replaces 27 random global memory reads per output voxel with a single cooperative smem load phase, and naturally expresses stencil operations (e.g. the 7-point 3D Laplacian) that only use a sparse subset of the 27 taps.

The primary motivation is **memory-bandwidth efficiency**: at `in_c = out_c = 1`, sparse convolution is completely memory-bound. The bottleneck is fetching neighbor feature values from global memory via NanoVDB hash lookups. By staging the 10×10×10 halo into shared memory cooperatively, each neighbor value is fetched from global memory exactly once per CTA regardless of how many output voxels reference it, and all subsequent accesses hit L1/smem.

---

## 2. Scope of Initial Implementation

The following constraints apply to the first implementation. Each is documented with the rationale and the expected path to relaxation.

| Constraint | Value | Rationale / Future Path |
|---|---|---|
| `in_channels` | 1 | Scalar-only; smem layout assumes scalar. Multi-channel requires smem of shape `[10][10][10][C]` and a vectorized load phase. |
| `out_channels` | 1 | Same. |
| Stencil radius R | 1 | Halo is exactly 10×10×10. R=2 would require 12×12×12=1728 smem slots; still feasible but a different kernel. |
| Stride | 1 | Stride > 1 changes the output leaf enumeration and the mapping from output voxel to input halo region. |
| Transposed convolution | Not supported | Requires swapping source/target roles and a separate halo computation. |
| Backward pass | Not implemented | Can be added later, either analytically (the backward of a stencil conv is another stencil conv with a transposed/flipped kernel) or by falling back to `GatherScatterDefault`. |
| Backend state | Stateless | No kmap or precomputed topology. The kernel receives source grid + target grid + features + weights at execute time and does all lookups on the fly. State can be added later (e.g. a precomputed list of active halo offsets for sparse stencils). |
| Weight tensor shape | `(1, 1, 3, 3, 3)` | Kept consistent with the existing API. Unused taps are set to zero by the caller; the kernel still performs the lookup and multiply, but the zero weight contributes nothing. A future sparse-stencil variant could store only the non-zero `(offset, weight)` pairs as backend state to skip the wasted lookups. |

---

## 3. Integration into `ConvolutionPlan`

### 3.1 Backend key

Selected via `expert_config={"backend": "stencil"}`.

### 3.2 `_StencilConvBackend` dataclass

```python
@dataclass(frozen=True)
class _StencilConvBackend:
    """Stencil convolution: CTA-per-leaf, smem halo staging, scalar only, stride 1.

    Stateless — no precomputed topology. The kernel receives source/target
    grid data and features at execute time.
    """
    pass
```

Because the dataclass is `frozen=True`, adding state later (e.g. a precomputed active-offset list for sparse stencils) is a non-breaking change: add a field with a default, populate it in `_build_backend`, access it in `execute`. No structural changes to the dispatch logic are needed.

### 3.3 `_Backend` union type (line 137)

```python
_Backend = _MatmulBackend | _DenseBackend | _GatherScatterBackend | _PredGatherIGemmBackend | _StencilConvBackend
```

### 3.4 Autograd function

No backward pass is implemented. The autograd `Function` wraps only `forward`; calling `backward` raises `NotImplementedError`. This is intentional and documented — the backend is designed for inference and non-differentiable preprocessing passes.

```python
class _StencilConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weights, source_grid_data, target_grid_data):
        return _fvdb_cpp.stencil_conv(features, weights, source_grid_data, target_grid_data)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "StencilConv backend does not support backward. "
            "Use 'gather_scatter' backend if gradients are required."
        )
```

### 3.5 Validation in `_build_backend`

The following checks are enforced at plan-creation time (i.e. when `ConvolutionPlan.from_grid_batch()` is called), not at execute time:

- `transposed == False` — transposed convolution is not supported.
- `_vec_is_all(stride, 1)` — only stride 1 is supported.
- `_vec_is_all(kernel_size, 3)` — kernel must be 3×3×3 (R=1 stencil).
- `channel_pairs` validation deferred to execute time (channels are not known at plan creation when `channel_pairs=_ANY_CHANNEL_PAIRS`).

Channel validation (`in_c == 1` and `out_c == 1`) is enforced in `execute()` before calling `_StencilConvFn.apply`.

### 3.6 Execute dispatch

```python
elif isinstance(backend, _StencilConvBackend):
    # Validate scalar channels
    if in_c != 1 or out_c != 1:
        raise ValueError(
            f"StencilConv backend requires in_channels=1 and out_channels=1, "
            f"got ({in_c}, {out_c})."
        )
    out_tensor = _StencilConvFn.apply(
        data.jdata,
        weights,
        _get_grid_data(self._source_grid),
        _get_grid_data(self._target_grid),
    )
    result = self._target_grid.jagged_like(out_tensor)
```

---

## 4. Algorithmic Design

### 4.1 Data layout and index convention

fVDB uses `NanoGrid<ValueOnIndex>` grids. Each active voxel in such a grid stores an integer index into a flat "sidecar" feature tensor. The NanoVDB layer returns **1-based** indices for active voxels and **0** (the background value) for inactive voxels.

fVDB shifts these by −1 before exposing them to the user, so:

| State | NanoVDB `getValue()` | fVDB feature index |
|---|---|---|
| Active voxel #k (0-based) | k+1 | k |
| Inactive voxel | 0 | −1 |

In the CUDA kernel, after querying the NanoVDB accessor, we apply the same −1 shift and store the result as `int64_t`. The sentinel for "inactive / no feature" is therefore **−1**. Guards throughout the kernel are `if (idx >= 0)`.

The input feature tensor has shape `(N_in,)` (scalar, float32). The output feature tensor has shape `(N_out,)`.

### 4.2 NanoVDB grids in play

- **Source grid**: `NanoGrid<ValueOnIndex>` representing the input domain. Used to populate the halo.
- **Target grid**: `NanoGrid<ValueOnIndex>` representing the output domain. Used to enumerate output voxels and map them to output feature indices.

The source and target grids may have different active sets (different sparsity patterns). The only requirement is that they share the same voxel size and origin (stride-1 convolution).

### 4.3 Kernel launch configuration

```
gridDim  = (num_target_leaves, 1, 1)
blockDim = (512, 1, 1)           // 8×8×8 = 512 threads per CTA
```

Each CTA is assigned one leaf node of the **target** grid. The CTA is responsible for computing the convolution output for all active voxels within that leaf.

### 4.4 Shared memory halo

A leaf node covers an axis-aligned 8×8×8 voxel region. With stencil radius R=1, each output voxel may depend on input voxels up to 1 step away in each dimension, giving a halo footprint of (8+2×1)³ = **10×10×10 = 1000 voxels**.

```cuda
__shared__ int64_t haloIndices[10][10][10];
```

Total shared memory: 1000 × 8 bytes = **8 KB per CTA**.

The halo region in world/index space relative to the target leaf origin `L = (Lx, Ly, Lz)`:

```
halo[i][j][k]  corresponds to input IJK = (Lx + i - 1,  Ly + j - 1,  Lz + k - 1)
                                           for i,j,k ∈ [0, 9]
```

The −1 offset accounts for the R=1 halo on the "low" side.

### 4.5 Halo population phase

512 threads cooperatively fill 1000 slots. We use a simple flat linearization:

```
slot s ∈ [0, 999]  →  (i, j, k) = (s / 100,  (s / 10) % 10,  s % 10)
```

Each thread handles `ceil(1000 / 512)` = 2 slots:

- **Pass 1** (all 512 threads): thread `t` handles slot `t`  (slots 0–511).
- **Pass 2** (first 488 threads): thread `t` handles slot `t + 512`  (slots 512–999). Threads 488–511 are idle in this pass.

For each slot:
1. Compute halo IJK = `(Lx + i - 1, Ly + j - 1, Lz + k - 1)`.
2. Query the **source** grid accessor: `uint64_t raw = sourceAccessor.getValue(ijk)`.
3. Apply the fVDB shift: `int64_t idx = static_cast<int64_t>(raw) - 1`.
4. Store: `haloIndices[i][j][k] = idx`.

After both passes: `__syncthreads()`.

### 4.6 Output voxel phase

Each thread owns exactly one voxel slot in the 8×8×8 leaf:

```
thread t  →  leaf-local offset (li, lj, lk) = (t / 64,  (t / 8) % 8,  t % 8)
           →  global IJK = (Lx + li,  Ly + lj,  Lz + lk)
```

Steps:
1. Query the **target** grid accessor for the output feature index:
   `int64_t outIdx = static_cast<int64_t>(targetAccessor.getValue(outIJK)) - 1`.
2. If `outIdx < 0`: this voxel is inactive in the target grid — **return early**, write nothing.
3. Accumulate the stencil:

```cuda
float sum = 0.0f;
for (int di = -1; di <= 1; ++di)
for (int dj = -1; dj <= 1; ++dj)
for (int dk = -1; dk <= 1; ++dk) {
    int64_t inIdx = haloIndices[li + di + 1][lj + dj + 1][lk + dk + 1];
    if (inIdx >= 0) {
        sum += weights[(di+1)*9 + (dj+1)*3 + (dk+1)] * inputFeatures[inIdx];
    }
}
```

4. Write: `outputFeatures[outIdx] = sum`.

### 4.7 Weights layout

The weight tensor has shape `(1, 1, 3, 3, 3)` (out_c, in_c, kx, ky, kz), matching the existing API. Inside the kernel, weights are accessed as a flat array of 27 floats via:

```
weight[offset]  where  offset = (di+1)*9 + (dj+1)*3 + (dk+1)
```

The weights tensor is small (27 × 4 bytes = 108 bytes) and will reside in L1/constant cache for the duration of the kernel — no explicit `__constant__` memory needed.

---

## 5. C++ Interface

### 5.1 Header — `src/fvdb/detail/ops/convolution/StencilConv.h`

```cpp
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/torch.h>
#include "fvdb/GridBatchImpl.h"   // adjust include path as needed

namespace fvdb {
namespace detail {

/// Forward-only scalar stencil convolution (in_c = out_c = 1, stride 1, R=1).
///
/// Launches one CTA per target leaf node. Each CTA stages the 10x10x10 input
/// halo into shared memory, then accumulates 27-tap stencil results for all
/// active output voxels in the leaf.
///
/// @param inputFeatures  (N_in,)      float32 scalar input features
/// @param weights        (1,1,3,3,3)  float32 stencil weights (27 taps)
/// @param sourceGrid     source NanoGrid<ValueOnIndex> (input domain)
/// @param targetGrid     target NanoGrid<ValueOnIndex> (output domain)
/// @returns              (N_out,) float32 scalar output features
torch::Tensor stencil_conv(
    const torch::Tensor&    inputFeatures,
    const torch::Tensor&    weights,
    const GridBatchImpl&    sourceGrid,
    const GridBatchImpl&    targetGrid
);

}  // namespace detail
}  // namespace fvdb
```

### 5.2 Implementation skeleton — `src/fvdb/detail/ops/convolution/StencilConv.cu`

```cpp
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "StencilConv.h"

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/cuda/CudaDeviceBuffer.h>

namespace fvdb {
namespace detail {

// ---------------------------------------------------------------------------
// CUDA kernel (skeleton — implementation TBD)
// ---------------------------------------------------------------------------

__global__ void stencilConvKernel(
    const float*   __restrict__ inputFeatures,   // (N_in,)
    const float*   __restrict__ weights,         // (27,) flat [di*9+dj*3+dk]
    /* NanoGrid<ValueOnIndex> source accessor */
    /* NanoGrid<ValueOnIndex> target accessor */
    float*         __restrict__ outputFeatures,  // (N_out,)
    int                         numTargetLeaves
)
{
    // TODO: implement
    //
    // Phase 1 — cooperative halo load:
    //   __shared__ int64_t haloIndices[10][10][10];
    //   Threads 0–511 fill slots 0–511; threads 0–487 fill slots 512–999.
    //   For each slot s → (i,j,k): query source accessor at (Lx+i-1, Ly+j-1, Lz+k-1),
    //   apply -1 shift, store as int64_t (-1 = inactive).
    //   __syncthreads();
    //
    // Phase 2 — stencil accumulation per output voxel:
    //   thread t owns leaf-local voxel (t/64, (t/8)%8, t%8).
    //   Query target accessor for output index; skip if inactive.
    //   27-tap MAC over haloIndices with guard (idx >= 0).
    //   Write to outputFeatures[outIdx].
}

// ---------------------------------------------------------------------------
// Host entry point
// ---------------------------------------------------------------------------

torch::Tensor stencil_conv(
    const torch::Tensor& inputFeatures,
    const torch::Tensor& weights,
    const GridBatchImpl& sourceGrid,
    const GridBatchImpl& targetGrid
)
{
    // TODO: implement
    //
    // 1. Validate shapes and dtypes (inputFeatures: (N_in,) float32,
    //    weights: (1,1,3,3,3) float32, both on CUDA).
    // 2. Allocate outputFeatures: torch::zeros({N_out}, inputFeatures.options()).
    // 3. Obtain NanoGrid device pointers from sourceGrid and targetGrid.
    // 4. Enumerate target leaf nodes; get numTargetLeaves.
    // 5. Launch: stencilConvKernel<<<numTargetLeaves, 512>>>(...)
    // 6. Return outputFeatures.

    TORCH_CHECK(false, "stencil_conv: not yet implemented");
    return {};
}

}  // namespace detail
}  // namespace fvdb
```

### 5.3 Python binding registration

In the appropriate binding file (e.g. `src/python/GridBatchOps.cpp` or a new `StencilConvBinding.cpp`):

```cpp
m.def("stencil_conv", &fvdb::detail::stencil_conv,
      "Forward-only scalar stencil convolution (CTA-per-leaf, smem halo, R=1).",
      py::arg("input_features"),
      py::arg("weights"),
      py::arg("source_grid"),
      py::arg("target_grid"));
```

And add a corresponding stub to `fvdb/_fvdb_cpp.pyi`:

```python
def stencil_conv(
    input_features: torch.Tensor,
    weights: torch.Tensor,
    source_grid: GridBatchImpl,
    target_grid: GridBatchImpl,
) -> torch.Tensor: ...
```

---

## 6. File Checklist

| File | Action |
|---|---|
| `fvdb/convolution_plan.py` | Add `_StencilConvBackend`, `_StencilConvFn`, branch in `_build_backend`, branch in `execute`, update `_Backend` union |
| `src/fvdb/detail/ops/convolution/StencilConv.h` | New — C++ function declaration |
| `src/fvdb/detail/ops/convolution/StencilConv.cu` | New — kernel skeleton + host entry point |
| `src/python/GridBatchOps.cpp` (or new binding file) | Register `stencil_conv` with pybind11 |
| `fvdb/_fvdb_cpp.pyi` | Add `stencil_conv` stub |
| `tests/unit/test_conv_stencil.py` | New — correctness tests (vs GatherScatterDefault on scalar data) |

---

## 7. Testing Strategy

Correctness is validated by comparing `StencilConv` output against `GatherScatterDefault` with equivalent scalar weights:

1. Create a random sparse `GridBatch` with a known active set.
2. Generate scalar input features `(N_in,)` and random weights `(1, 1, 3, 3, 3)`.
3. Run via `expert_config={"backend": "stencil"}`.
4. Run via `expert_config={"backend": "gather_scatter"}` with the same weights (shaped `(1, 1, 3, 3, 3)`).
5. Assert `torch.allclose(stencil_output, gs_output, atol=1e-5)`.

Additionally test:
- Output is zero for output voxels with no active input neighbors.
- Sparse stencil (e.g. 7-pt Laplacian weights with zeros at corners/edges) matches GS reference.
- Invalid-configuration rejection: `in_c != 1`, stride != 1, transposed, non-3×3×3 kernel all raise `ValueError` at plan-creation time.

---

## 8. Future Work

In rough priority order:

1. **Backward pass** — The backward of a scalar stencil convolution is another stencil convolution with the kernel flipped (`weights[:, :, ::-1, ::-1, ::-1]`) applied from the output gradient grid to the input gradient. This maps cleanly to the same CTA-per-leaf design. Alternatively, fall back to `GatherScatterDefault` backward as `PredGatherIGemm` does.

2. **Sparse stencil support** — Store the list of non-zero `(offset_ijk, weight)` pairs as backend state (computed once at plan-creation from the weight tensor). The kernel iterates over only the active taps, eliminating NanoVDB lookups for zero-weight offsets. For a 7-pt Laplacian, this reduces lookups per output voxel from 27 to 7.

3. **Multi-channel extension** — Extend the smem layout to `haloIndices[10][10][10]` (unchanged) but load `C` features per halo voxel. The inner loop becomes a vector dot-product. The bound on `C` is determined by the remaining smem budget after the index array.

4. **Stride > 1** — Changes the output leaf enumeration (target leaves may not align with source leaves) and the halo size calculation.

5. **Larger stencil radii** — R=2 → 12×12×12=1728 slots × 8 bytes = 13.5 KB smem; still feasible on most hardware. Requires parameterizing the halo size and the weight tensor shape.
