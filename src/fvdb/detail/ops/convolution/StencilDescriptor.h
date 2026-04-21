// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file StencilDescriptor.h
/// @brief Compile-time stencil descriptors for StencilConv.
///
/// A stencil descriptor is a type whose nested `Taps` member is a
/// `std::tuple` of `StencilPoint<DI, DJ, DK>` types. The CTA-per-leaf
/// kernel template `stencilConvKernel<StencilT>` unrolls its
/// accumulation loop over `StencilT::Taps`, generating one FMA per tap
/// at compile time.
///
/// Weights remain shape [1,1,3,3,3]; only the slots corresponding to
/// taps in the chosen descriptor are read. The Python plan layer is
/// expected to validate (at execute time) that off-stencil weight
/// positions are zero, so a sparse-stencil forward pass gives the same
/// numerical result as the dense 27-tap path on Laplace7-shaped
/// weights.

#ifndef FVDB_DETAIL_OPS_CONVOLUTION_STENCILDESCRIPTOR_H
#define FVDB_DETAIL_OPS_CONVOLUTION_STENCILDESCRIPTOR_H

#include <cstddef>
#include <tuple>
#include <utility>

namespace fvdb {
namespace detail {
namespace ops {

/// Compile-time 3D stencil tap offset.
template <int DI, int DJ, int DK>
struct StencilPoint {
    static constexpr int di = DI;
    static constexpr int dj = DJ;
    static constexpr int dk = DK;
};

namespace detail_stencil {

template <typename TupsT, int DI, int DJ, int DK, std::size_t... Is>
constexpr int
findIndexImpl(std::index_sequence<Is...>) {
    int result = -1;
    ((std::tuple_element_t<Is, TupsT>::di == DI &&
              std::tuple_element_t<Is, TupsT>::dj == DJ &&
              std::tuple_element_t<Is, TupsT>::dk == DK && result < 0
          ? (result = static_cast<int>(Is))
          : 0),
     ...);
    return result;
}

} // namespace detail_stencil

/// Compile-time inverse map: returns the first tuple slot in `TupsT`
/// whose (DI, DJ, DK) matches, or -1 if not present.
template <typename TupsT, int DI, int DJ, int DK>
constexpr int
findIndex() {
    return detail_stencil::findIndexImpl<TupsT, DI, DJ, DK>(
        std::make_index_sequence<std::tuple_size_v<TupsT>>{});
}

/// Dense 3x3x3 27-tap stencil — matches the historic kernel behavior.
struct Dense27Stencil {
    using Taps = std::tuple<
        StencilPoint<-1, -1, -1>, StencilPoint<-1, -1,  0>, StencilPoint<-1, -1, +1>,
        StencilPoint<-1,  0, -1>, StencilPoint<-1,  0,  0>, StencilPoint<-1,  0, +1>,
        StencilPoint<-1, +1, -1>, StencilPoint<-1, +1,  0>, StencilPoint<-1, +1, +1>,
        StencilPoint< 0, -1, -1>, StencilPoint< 0, -1,  0>, StencilPoint< 0, -1, +1>,
        StencilPoint< 0,  0, -1>, StencilPoint< 0,  0,  0>, StencilPoint< 0,  0, +1>,
        StencilPoint< 0, +1, -1>, StencilPoint< 0, +1,  0>, StencilPoint< 0, +1, +1>,
        StencilPoint<+1, -1, -1>, StencilPoint<+1, -1,  0>, StencilPoint<+1, -1, +1>,
        StencilPoint<+1,  0, -1>, StencilPoint<+1,  0,  0>, StencilPoint<+1,  0, +1>,
        StencilPoint<+1, +1, -1>, StencilPoint<+1, +1,  0>, StencilPoint<+1, +1, +1>>;
};

/// 7-point 3D Laplacian stencil — center + 6 face neighbors.
struct Laplace3DStencil {
    using Taps = std::tuple<
        StencilPoint< 0,  0,  0>,
        StencilPoint<-1,  0,  0>, StencilPoint<+1,  0,  0>,
        StencilPoint< 0, -1,  0>, StencilPoint< 0, +1,  0>,
        StencilPoint< 0,  0, -1>, StencilPoint< 0,  0, +1>>;
};

/// Runtime tag to pick a compile-time stencil specialization.
enum class StencilKind : int {
    Dense27  = 0,
    Laplace7 = 1,
};

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_STENCILDESCRIPTOR_H
