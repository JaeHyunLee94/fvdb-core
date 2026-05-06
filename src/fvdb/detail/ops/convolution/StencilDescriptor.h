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

/// Compile-time 3D stencil tap offset (scalar in/out).
template <int DI, int DJ, int DK>
struct StencilPoint {
    static constexpr int di = DI;
    static constexpr int dj = DJ;
    static constexpr int dk = DK;
};

/// Compile-time multi-channel stencil tap. Each tap carries which input
/// and output channel pair it connects, plus the spatial offset.
///   sum[out=OC] += weights[OC, IC, DI+1, DJ+1, DK+1] * halo[..., IC]
template <int OC, int IC, int DI, int DJ, int DK>
struct ChannelTap {
    static constexpr int oc = OC;
    static constexpr int ic = IC;
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

/// 3D divergence stencil: input is a 3-component vector field, output is
/// a scalar. Six taps total, each connecting one input channel to the
/// single output channel with a central-difference along that channel's axis.
///
/// Weight tensor shape: [out_c=1, in_c=3, 3, 3, 3]. On-stencil positions:
///   weights[0, 0, 0|2, 1, 1]  →  ∂Fx/∂x
///   weights[0, 1, 1, 0|2, 1]  →  ∂Fy/∂y
///   weights[0, 2, 1, 1, 0|2]  →  ∂Fz/∂z
struct Divergence3DStencil {
    static constexpr int kInChannels  = 3;
    static constexpr int kOutChannels = 1;
    using Taps = std::tuple<
        ChannelTap<0, 0, -1,  0,  0>, ChannelTap<0, 0, +1,  0,  0>,  // ∂Fx/∂x
        ChannelTap<0, 1,  0, -1,  0>, ChannelTap<0, 1,  0, +1,  0>,  // ∂Fy/∂y
        ChannelTap<0, 2,  0,  0, -1>, ChannelTap<0, 2,  0,  0, +1>>; // ∂Fz/∂z
};

/// 3D gradient stencil: input is a scalar field, output is a 3-component
/// vector. Six taps total; each output channel receives a central-difference
/// pair along its corresponding axis.
///
/// Weight tensor shape: [out_c=3, in_c=1, 3, 3, 3]. On-stencil positions:
///   weights[0, 0, 0|2, 1, 1]  →  ∂/∂x
///   weights[1, 0, 1, 0|2, 1]  →  ∂/∂y
///   weights[2, 0, 1, 1, 0|2]  →  ∂/∂z
struct Gradient3DStencil {
    static constexpr int kInChannels  = 1;
    static constexpr int kOutChannels = 3;
    using Taps = std::tuple<
        ChannelTap<0, 0, -1,  0,  0>, ChannelTap<0, 0, +1,  0,  0>,  // ∂/∂x → out ch 0
        ChannelTap<1, 0,  0, -1,  0>, ChannelTap<1, 0,  0, +1,  0>,  // ∂/∂y → out ch 1
        ChannelTap<2, 0,  0,  0, -1>, ChannelTap<2, 0,  0,  0, +1>>; // ∂/∂z → out ch 2
};

/// MAC-grid divergence: forward differences. Used when each velocity
/// component lives on the +face of the cell (u on +x face, etc.), so
/// div(u) at the cell center is u[i+1] - u[i] etc.
///
/// Weight tensor shape: [1, 3, 3, 3, 3]. On-stencil positions:
///   weights[0, 0, 1|2, 1, 1]  (ic=0, di ∈ {0, +1})  →  ∂Fx/∂x
///   weights[0, 1, 1, 1|2, 1]  (ic=1, dj ∈ {0, +1})  →  ∂Fy/∂y
///   weights[0, 2, 1, 1, 1|2]  (ic=2, dk ∈ {0, +1})  →  ∂Fz/∂z
struct MacDivergence3DStencil {
    static constexpr int kInChannels  = 3;
    static constexpr int kOutChannels = 1;
    using Taps = std::tuple<
        ChannelTap<0, 0,  0,  0,  0>, ChannelTap<0, 0, +1,  0,  0>,  // axis x, ic=0
        ChannelTap<0, 1,  0,  0,  0>, ChannelTap<0, 1,  0, +1,  0>,  // axis y, ic=1
        ChannelTap<0, 2,  0,  0,  0>, ChannelTap<0, 2,  0,  0, +1>>; // axis z, ic=2
};

/// MAC-grid gradient: backward differences. Pairs with MacDivergence so
/// the resulting Laplacian (div ∘ grad) is the standard 7-point stencil.
///
/// Weight tensor shape: [3, 1, 3, 3, 3]. On-stencil positions:
///   weights[0, 0, 0|1, 1, 1]  (oc=0, di ∈ {-1, 0})  →  ∂/∂x → out 0
///   weights[1, 0, 1, 0|1, 1]  (oc=1, dj ∈ {-1, 0})  →  ∂/∂y → out 1
///   weights[2, 0, 1, 1, 0|1]  (oc=2, dk ∈ {-1, 0})  →  ∂/∂z → out 2
struct MacGradient3DStencil {
    static constexpr int kInChannels  = 1;
    static constexpr int kOutChannels = 3;
    using Taps = std::tuple<
        ChannelTap<0, 0, -1,  0,  0>, ChannelTap<0, 0,  0,  0,  0>,  // ∂/∂x → out 0
        ChannelTap<1, 0,  0, -1,  0>, ChannelTap<1, 0,  0,  0,  0>,  // ∂/∂y → out 1
        ChannelTap<2, 0,  0,  0, -1>, ChannelTap<2, 0,  0,  0,  0>>; // ∂/∂z → out 2
};

/// Runtime tag to pick a compile-time stencil specialization.
enum class StencilKind : int {
    Dense27       = 0,  // in=1, out=1, 27 taps
    Laplace7      = 1,  // in=1, out=1, 7 taps
    Divergence    = 2,  // in=3, out=1, 6 taps  (central differences)
    Gradient      = 3,  // in=1, out=3, 6 taps  (central differences)
    MacDivergence = 4,  // in=3, out=1, 6 taps  (forward differences, MAC grid)
    MacGradient   = 5,  // in=1, out=3, 6 taps  (backward differences, MAC grid)
};

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_STENCILDESCRIPTOR_H
