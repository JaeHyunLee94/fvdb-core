// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file StencilConv.h
/// @brief Forward-only scalar stencil sparse convolution.
///
/// Launches one CTA per target leaf node.  Each CTA stages the 10x10x10 input
/// halo of the leaf into shared memory via cooperative NanoVDB lookups, then
/// each of the 512 threads in the CTA accumulates a 27-tap multiply-add for
/// its assigned output voxel and writes the result.
///
/// Constraints (initial implementation):
///   - Forward only, no gradient.
///   - in_channels = out_channels = 1 (scalar features).
///   - Kernel size 3x3x3, radius R=1.
///   - Stride 1, not transposed.
///   - float32 tensors only, on CUDA.
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_STENCILCONV_H
#define FVDB_DETAIL_OPS_CONVOLUTION_STENCILCONV_H

#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/StencilDescriptor.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Forward pass of scalar stencil sparse convolution.
///
/// @param inputFeatures  Input features, shape [N_in] or [N_in, 1], float32, on CUDA.
/// @param weights        Kernel weights, shape [1, 1, 3, 3, 3], float32, on CUDA.
/// @param sourceGrid     Grid batch for the input (feature) voxels.
/// @param targetGrid     Grid batch for the output voxels.
/// @param kind           Compile-time stencil specialization (Dense27 or Laplace7).
/// @return               Output features, shape [N_out, 1], float32.
torch::Tensor stencilSparseConv(const torch::Tensor &inputFeatures,
                                const torch::Tensor &weights,
                                const GridBatchImpl &sourceGrid,
                                const GridBatchImpl &targetGrid,
                                StencilKind          kind = StencilKind::Dense27);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_STENCILCONV_H
