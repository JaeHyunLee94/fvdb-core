# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Test the StencilConv sparse convolution backend.

The StencilConv backend has strict constraints:
  - CUDA only
  - float32 only
  - Forward pass only (no backward, no transpose)
  - 3x3x3 kernel (R=1 stencil)
  - Stride 1
  - in_channels = out_channels = 1 (scalar)
  - Batch size 1

Correctness is validated by cross-backend comparison against the
GatherScatterDefault backend with equivalent scalar weights, per §7 of
stencil_conv_backend.md.
"""

import unittest

import torch
from fvdb.convolution_plan import _StencilConvBackend
from fvdb.utils.tests import generate_hermit_impulses_dense
from fvdb.utils.tests.convolution_utils import (
    create_grid_from_coords,
    get_cluster_edge_aligned,
)

from fvdb import ConvolutionPlan, JaggedTensor

# =============================================================================
# Configuration
# =============================================================================

STENCIL_CONFIG: dict = {"backend": "stencil"}
STENCIL_LAPLACE7_CONFIG: dict = {"backend": "stencil", "stencil": "laplace7"}
STENCIL_DIVERGENCE_CONFIG: dict = {"backend": "stencil", "stencil": "divergence"}
STENCIL_GRADIENT_CONFIG: dict = {"backend": "stencil", "stencil": "gradient"}
GS_CONFIG: dict = {"backend": "gather_scatter"}

# Scalar fp32 — strict tolerance since both backends do the same accumulation
# in fp32, just in different orders.
RTOL = 1e-5
ATOL = 1e-5


# =============================================================================
# Test Class
# =============================================================================


class TestConvStencil(unittest.TestCase):
    """Forward-only tests for the StencilConv backend."""

    DEVICE = torch.device("cuda", 0)
    DTYPE = torch.float32
    KERNEL_SIZE = (3, 3, 3)
    VOLUME_SHAPE = (71, 34, 58)
    NUM_CANDIDATES = 1000

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("StencilConv requires CUDA")
        torch.manual_seed(2024)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _run_both_backends(self, grid, features, weights):
        """Execute the same (grid, features, weights) via stencil and gather_scatter."""
        dst_grid = grid.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)

        stencil_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid,
            target_grid=dst_grid,
            expert_config=STENCIL_CONFIG,
        )
        self.assertIsInstance(stencil_plan._backend, _StencilConvBackend)

        gs_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid,
            target_grid=dst_grid,
            expert_config=GS_CONFIG,
        )

        stencil_out = stencil_plan.execute(features, weights)
        gs_out = gs_plan.execute(features, weights)
        return stencil_out, gs_out

    # -------------------------------------------------------------------------
    # Basic cross-backend correctness
    # -------------------------------------------------------------------------

    def test_matches_gather_scatter_single_impulse(self):
        """One active voxel at a known coordinate, random dense stencil."""
        coord = torch.tensor([[5, 5, 5]], device=self.DEVICE, dtype=torch.int32)
        grid = create_grid_from_coords(coord, self.DEVICE)

        features = JaggedTensor(torch.ones((1, 1), device=self.DEVICE, dtype=self.DTYPE))
        weights = torch.randn((1, 1, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)

        stencil_out, gs_out = self._run_both_backends(grid, features, weights)

        torch.testing.assert_close(stencil_out.jdata, gs_out.jdata, rtol=RTOL, atol=ATOL)

    def test_matches_gather_scatter_dense_cluster(self):
        """Dense cluster (many overlapping halos) with random stencil weights."""
        cluster = get_cluster_edge_aligned(self.KERNEL_SIZE, self.DEVICE)
        grid = create_grid_from_coords(cluster, self.DEVICE)
        n = len(cluster)

        features = JaggedTensor(torch.randn((n, 1), device=self.DEVICE, dtype=self.DTYPE))
        weights = torch.randn((1, 1, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)

        stencil_out, gs_out = self._run_both_backends(grid, features, weights)
        torch.testing.assert_close(stencil_out.jdata, gs_out.jdata, rtol=RTOL, atol=ATOL)

    def test_matches_gather_scatter_many_impulses(self):
        """Many non-overlapping impulses spread across a large volume."""
        impulse_coords, _ = generate_hermit_impulses_dense(
            num_candidates=self.NUM_CANDIDATES,
            volume_shape=self.VOLUME_SHAPE,
            kernel_size=self.KERNEL_SIZE,
            impulse_value=1,
            dtype=self.DTYPE,
            device=self.DEVICE,
        )
        self.assertGreater(len(impulse_coords), 0)

        grid = create_grid_from_coords(impulse_coords, self.DEVICE)
        n = grid.total_voxels

        features = JaggedTensor(torch.randn((n, 1), device=self.DEVICE, dtype=self.DTYPE))
        weights = torch.randn((1, 1, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)

        stencil_out, gs_out = self._run_both_backends(grid, features, weights)
        torch.testing.assert_close(stencil_out.jdata, gs_out.jdata, rtol=RTOL, atol=ATOL)

    def test_matches_gather_scatter_7pt_laplacian(self):
        """Sparse 7-point 3D Laplacian weights (corners/edges zero)."""
        cluster = get_cluster_edge_aligned(self.KERNEL_SIZE, self.DEVICE)
        grid = create_grid_from_coords(cluster, self.DEVICE)
        n = len(cluster)

        # 7-point Laplacian: center = -6, 6 face neighbors = 1, rest = 0
        weights = torch.zeros((1, 1, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)
        weights[0, 0, 1, 1, 1] = -6.0
        for (di, dj, dk) in [(0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)]:
            weights[0, 0, di, dj, dk] = 1.0

        features = JaggedTensor(torch.randn((n, 1), device=self.DEVICE, dtype=self.DTYPE))

        stencil_out, gs_out = self._run_both_backends(grid, features, weights)
        torch.testing.assert_close(stencil_out.jdata, gs_out.jdata, rtol=RTOL, atol=ATOL)

    # -------------------------------------------------------------------------
    # Shape and zero-fill behavior
    # -------------------------------------------------------------------------

    def test_output_shape_matches_target_grid(self):
        """Output jdata first dim equals target grid total_voxels; channel dim == 1."""
        cluster = get_cluster_edge_aligned(self.KERNEL_SIZE, self.DEVICE)
        grid = create_grid_from_coords(cluster, self.DEVICE)
        dst_grid = grid.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)

        features = JaggedTensor(torch.randn((len(cluster), 1), device=self.DEVICE, dtype=self.DTYPE))
        weights = torch.randn((1, 1, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)

        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid,
            target_grid=dst_grid,
            expert_config=STENCIL_CONFIG,
        )
        out = plan.execute(features, weights)
        self.assertEqual(out.jdata.shape, (dst_grid.total_voxels, 1))

    # -------------------------------------------------------------------------
    # Validation — wrong configs rejected at plan-creation time
    # -------------------------------------------------------------------------

    def test_rejects_transposed(self):
        grid = create_grid_from_coords(
            get_cluster_edge_aligned(self.KERNEL_SIZE, self.DEVICE), self.DEVICE
        )
        dst_grid = grid.conv_grid(self.KERNEL_SIZE, 1)
        with self.assertRaises(ValueError):
            ConvolutionPlan.from_grid_batch_transposed(
                kernel_size=self.KERNEL_SIZE,
                stride=1,
                source_grid=grid,
                target_grid=dst_grid,
                expert_config=STENCIL_CONFIG,
            )

    def test_rejects_stride_2(self):
        grid = create_grid_from_coords(
            get_cluster_edge_aligned(self.KERNEL_SIZE, self.DEVICE), self.DEVICE
        )
        with self.assertRaises(ValueError):
            ConvolutionPlan.from_grid_batch(
                kernel_size=self.KERNEL_SIZE,
                stride=2,
                source_grid=grid,
                expert_config=STENCIL_CONFIG,
            )

    def test_rejects_non_3x3x3_kernel(self):
        grid = create_grid_from_coords(
            get_cluster_edge_aligned((5, 5, 5), self.DEVICE), self.DEVICE
        )
        with self.assertRaises(ValueError):
            ConvolutionPlan.from_grid_batch(
                kernel_size=5,
                stride=1,
                source_grid=grid,
                expert_config=STENCIL_CONFIG,
            )

    # -------------------------------------------------------------------------
    # Validation — non-scalar channels rejected at execute time
    # -------------------------------------------------------------------------

    def test_rejects_multi_channel_at_execute(self):
        cluster = get_cluster_edge_aligned(self.KERNEL_SIZE, self.DEVICE)
        grid = create_grid_from_coords(cluster, self.DEVICE)
        n = len(cluster)

        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid,
            expert_config=STENCIL_CONFIG,
        )

        features = JaggedTensor(torch.randn((n, 4), device=self.DEVICE, dtype=self.DTYPE))
        weights = torch.randn((4, 4, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)
        with self.assertRaises(ValueError):
            plan.execute(features, weights)

    # -------------------------------------------------------------------------
    # Laplace7 specialization
    # -------------------------------------------------------------------------

    LAPLACE7_ON_STENCIL = [
        (1, 1, 1),
        (0, 1, 1), (2, 1, 1),
        (1, 0, 1), (1, 2, 1),
        (1, 1, 0), (1, 1, 2),
    ]

    def _make_laplace7_weights(self) -> torch.Tensor:
        """Build a strict 7-point Laplacian weight tensor (off-stencil exactly 0)."""
        w = torch.zeros((1, 1, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)
        w[0, 0, 1, 1, 1] = -6.0
        for i, j, k in self.LAPLACE7_ON_STENCIL[1:]:
            w[0, 0, i, j, k] = 1.0
        return w

    def _make_random_laplace7_weights(self) -> torch.Tensor:
        """Random on-stencil values, off-stencil positions exactly 0."""
        w = torch.zeros((1, 1, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)
        for i, j, k in self.LAPLACE7_ON_STENCIL:
            w[0, 0, i, j, k] = torch.randn((), device=self.DEVICE, dtype=self.DTYPE)
        return w

    def test_laplace7_matches_dense27_standard_laplacian(self):
        """Laplace7 kernel with classical [-6, 1, 1, 1, 1, 1, 1] coefficients matches Dense27."""
        cluster = get_cluster_edge_aligned(self.KERNEL_SIZE, self.DEVICE)
        grid = create_grid_from_coords(cluster, self.DEVICE)
        dst_grid = grid.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        n = len(cluster)

        features = JaggedTensor(torch.randn((n, 1), device=self.DEVICE, dtype=self.DTYPE))
        weights = self._make_laplace7_weights()

        dense_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid,
            target_grid=dst_grid,
            expert_config=STENCIL_CONFIG,
        )
        laplace_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid,
            target_grid=dst_grid,
            expert_config=STENCIL_LAPLACE7_CONFIG,
        )
        dense_out = dense_plan.execute(features, weights)
        laplace_out = laplace_plan.execute(features, weights)

        # Both kernels compute the same math when only 7 weight positions are nonzero,
        # so results should be bit-identical.
        torch.testing.assert_close(laplace_out.jdata, dense_out.jdata, rtol=0.0, atol=0.0)

    def test_laplace7_matches_dense27_random_on_stencil_weights(self):
        """Laplace7 matches Dense27 for any weight with off-stencil positions zero."""
        impulse_coords, _ = generate_hermit_impulses_dense(
            num_candidates=self.NUM_CANDIDATES,
            volume_shape=self.VOLUME_SHAPE,
            kernel_size=self.KERNEL_SIZE,
            impulse_value=1,
            dtype=self.DTYPE,
            device=self.DEVICE,
        )
        grid = create_grid_from_coords(impulse_coords, self.DEVICE)
        dst_grid = grid.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        n = grid.total_voxels
        features = JaggedTensor(torch.randn((n, 1), device=self.DEVICE, dtype=self.DTYPE))
        weights = self._make_random_laplace7_weights()

        dense_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid, target_grid=dst_grid,
            expert_config=STENCIL_CONFIG,
        )
        laplace_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid, target_grid=dst_grid,
            expert_config=STENCIL_LAPLACE7_CONFIG,
        )
        torch.testing.assert_close(
            laplace_plan.execute(features, weights).jdata,
            dense_plan.execute(features, weights).jdata,
            rtol=0.0, atol=0.0,
        )

    def test_laplace7_matches_gather_scatter(self):
        """Laplace7 result equals gather_scatter on a pure 7-point Laplacian."""
        cluster = get_cluster_edge_aligned(self.KERNEL_SIZE, self.DEVICE)
        grid = create_grid_from_coords(cluster, self.DEVICE)
        dst_grid = grid.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        n = len(cluster)

        features = JaggedTensor(torch.randn((n, 1), device=self.DEVICE, dtype=self.DTYPE))
        weights = self._make_laplace7_weights()

        laplace_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid, target_grid=dst_grid,
            expert_config=STENCIL_LAPLACE7_CONFIG,
        )
        gs_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid, target_grid=dst_grid,
            expert_config=GS_CONFIG,
        )
        torch.testing.assert_close(
            laplace_plan.execute(features, weights).jdata,
            gs_plan.execute(features, weights).jdata,
            rtol=RTOL, atol=ATOL,
        )

    def test_laplace7_rejects_nonzero_off_stencil_weights(self):
        """Passing weights with nonzero off-stencil positions must raise at execute time."""
        cluster = get_cluster_edge_aligned(self.KERNEL_SIZE, self.DEVICE)
        grid = create_grid_from_coords(cluster, self.DEVICE)
        n = len(cluster)

        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid,
            expert_config=STENCIL_LAPLACE7_CONFIG,
        )
        # A fully-random 3x3x3 tensor has nonzero corners/edges, so it should fail.
        features = JaggedTensor(torch.randn((n, 1), device=self.DEVICE, dtype=self.DTYPE))
        weights = torch.randn((1, 1, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)
        with self.assertRaises(ValueError) as cm:
            plan.execute(features, weights)
        self.assertIn("laplace7", str(cm.exception).lower())

    def test_laplace7_unknown_stencil_name_rejected(self):
        """expert_config={'stencil': '<unknown>'} is rejected at plan creation."""
        cluster = get_cluster_edge_aligned(self.KERNEL_SIZE, self.DEVICE)
        grid = create_grid_from_coords(cluster, self.DEVICE)
        with self.assertRaises(ValueError):
            ConvolutionPlan.from_grid_batch(
                kernel_size=self.KERNEL_SIZE, stride=1,
                source_grid=grid,
                expert_config={"backend": "stencil", "stencil": "not_a_real_stencil"},
            )

    # -------------------------------------------------------------------------
    # Correctness — source_grid != target_grid
    # -------------------------------------------------------------------------

    def _run_both_with_explicit_grids(self, src_grid, tgt_grid, features, weights):
        """Run stencil + gather_scatter with an explicit, possibly unrelated, target grid."""
        stencil_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=src_grid,
            target_grid=tgt_grid,
            expert_config=STENCIL_CONFIG,
        )
        self.assertIsInstance(stencil_plan._backend, _StencilConvBackend)

        gs_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=src_grid,
            target_grid=tgt_grid,
            expert_config=GS_CONFIG,
        )
        return stencil_plan.execute(features, weights), gs_plan.execute(features, weights)

    def test_matches_gs_disjoint_source_target(self):
        """Source and target are different topologies that only partially overlap."""
        # Source: solid 5x5x5 block near (5,5,5).
        src_coords = torch.tensor(
            [[5 + i, 5 + j, 5 + k] for i in range(5) for j in range(5) for k in range(5)],
            device=self.DEVICE,
            dtype=torch.int32,
        )
        # Target: shifted + slightly larger block. Many target voxels have no active
        # source within their 3x3x3 halo -> output must still be zero for those.
        tgt_coords = torch.tensor(
            [[7 + i, 7 + j, 7 + k] for i in range(6) for j in range(6) for k in range(6)],
            device=self.DEVICE,
            dtype=torch.int32,
        )
        src_grid = create_grid_from_coords(src_coords, self.DEVICE)
        tgt_grid = create_grid_from_coords(tgt_coords, self.DEVICE)

        features = JaggedTensor(
            torch.randn((src_grid.total_voxels, 1), device=self.DEVICE, dtype=self.DTYPE)
        )
        weights = torch.randn((1, 1, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)

        stencil_out, gs_out = self._run_both_with_explicit_grids(
            src_grid, tgt_grid, features, weights
        )
        torch.testing.assert_close(stencil_out.jdata, gs_out.jdata, rtol=RTOL, atol=ATOL)

    def test_matches_gs_target_subset_of_source(self):
        """Target is a strict subset of the source grid."""
        cluster = get_cluster_edge_aligned(self.KERNEL_SIZE, self.DEVICE)
        src_grid = create_grid_from_coords(cluster, self.DEVICE)
        # Pick every other voxel of the source cluster as the target.
        tgt_coords = cluster[::2].contiguous()
        tgt_grid = create_grid_from_coords(tgt_coords, self.DEVICE)

        features = JaggedTensor(
            torch.randn((src_grid.total_voxels, 1), device=self.DEVICE, dtype=self.DTYPE)
        )
        weights = torch.randn((1, 1, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)

        stencil_out, gs_out = self._run_both_with_explicit_grids(
            src_grid, tgt_grid, features, weights
        )
        torch.testing.assert_close(stencil_out.jdata, gs_out.jdata, rtol=RTOL, atol=ATOL)

    # -------------------------------------------------------------------------
    # Correctness — complex sparse topologies
    # -------------------------------------------------------------------------

    def test_matches_gs_hollow_cube_shell(self):
        """Active voxels form only the surface of a cube (hollow shell)."""
        size = 12
        base = 4
        coords = []
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    on_surface = (
                        i in (0, size - 1) or j in (0, size - 1) or k in (0, size - 1)
                    )
                    if on_surface:
                        coords.append([base + i, base + j, base + k])
        coords_t = torch.tensor(coords, device=self.DEVICE, dtype=torch.int32)
        grid = create_grid_from_coords(coords_t, self.DEVICE)
        n = grid.total_voxels

        features = JaggedTensor(torch.randn((n, 1), device=self.DEVICE, dtype=self.DTYPE))
        weights = torch.randn((1, 1, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)

        stencil_out, gs_out = self._run_both_backends(grid, features, weights)
        torch.testing.assert_close(stencil_out.jdata, gs_out.jdata, rtol=RTOL, atol=ATOL)

    def test_matches_gs_leaf_boundary_spanning(self):
        """Voxels placed deliberately on 8x8x8 leaf boundaries to stress halo lookups."""
        # Leaf size is 8. Place voxels straddling boundaries at multiples of 8.
        coords_list = []
        for base_i in (0, 8, 16):
            for base_j in (0, 8, 16):
                for base_k in (0, 8, 16):
                    # Corner, edge, face, and interior samples for each leaf-aligned anchor.
                    samples = [
                        (0, 0, 0),  # corner of a leaf
                        (7, 7, 7),  # opposite corner (same leaf)
                        (0, 7, 0),  # an edge of the leaf
                        (3, 4, 5),  # interior
                    ]
                    for di, dj, dk in samples:
                        coords_list.append([base_i + di, base_j + dj, base_k + dk])
        coords_t = torch.tensor(coords_list, device=self.DEVICE, dtype=torch.int32)
        # Unique-ify in case of duplicates across blocks.
        coords_t = torch.unique(coords_t, dim=0)
        grid = create_grid_from_coords(coords_t, self.DEVICE)
        n = grid.total_voxels

        features = JaggedTensor(torch.randn((n, 1), device=self.DEVICE, dtype=self.DTYPE))
        weights = torch.randn((1, 1, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)

        stencil_out, gs_out = self._run_both_backends(grid, features, weights)
        torch.testing.assert_close(stencil_out.jdata, gs_out.jdata, rtol=RTOL, atol=ATOL)

    def test_matches_gs_random_dropout_block(self):
        """Dense block with ~60% random dropout — irregular topology inside each leaf."""
        torch.manual_seed(17)
        size = 16
        base = 4
        full = torch.tensor(
            [[base + i, base + j, base + k] for i in range(size) for j in range(size) for k in range(size)],
            device=self.DEVICE,
            dtype=torch.int32,
        )
        keep = torch.rand(full.shape[0], device=self.DEVICE) > 0.6
        coords_t = full[keep]
        self.assertGreater(len(coords_t), 0)
        grid = create_grid_from_coords(coords_t, self.DEVICE)
        n = grid.total_voxels

        features = JaggedTensor(torch.randn((n, 1), device=self.DEVICE, dtype=self.DTYPE))
        weights = torch.randn((1, 1, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)

        stencil_out, gs_out = self._run_both_backends(grid, features, weights)
        torch.testing.assert_close(stencil_out.jdata, gs_out.jdata, rtol=RTOL, atol=ATOL)

    def test_matches_gs_thin_diagonal_curve(self):
        """Sparse 1-voxel-thick diagonal curve — each voxel has few active neighbors."""
        length = 40
        base = 3
        coords = []
        for t in range(length):
            # Simple diagonal-ish curve with a wobble to exercise neighbor lookups.
            coords.append([base + t, base + (t // 2), base + ((t * 3) % 7)])
        coords_t = torch.tensor(coords, device=self.DEVICE, dtype=torch.int32)
        coords_t = torch.unique(coords_t, dim=0)
        grid = create_grid_from_coords(coords_t, self.DEVICE)
        n = grid.total_voxels

        features = JaggedTensor(torch.randn((n, 1), device=self.DEVICE, dtype=self.DTYPE))
        weights = torch.randn((1, 1, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)

        stencil_out, gs_out = self._run_both_backends(grid, features, weights)
        torch.testing.assert_close(stencil_out.jdata, gs_out.jdata, rtol=RTOL, atol=ATOL)

    # -------------------------------------------------------------------------
    # Performance — effective bandwidth vs default backend (A6000 768 GB/s)
    # -------------------------------------------------------------------------

    # Effective bandwidth accounts for the minimum data traffic: one read of
    # every active source voxel's feature and one write of every active target
    # voxel's feature, at 4 bytes per float32 element. Actual device traffic
    # is higher (halo gathers re-read neighbors, grid index lookups hit the
    # NanoVDB accessor tree), but this "useful-data" bandwidth is the metric
    # most directly comparable to HBM peak.
    A6000_PEAK_GBPS = 768.0
    BYTES_PER_FLOAT32 = 4

    def _benchmark_plan(self, plan, features, weights, warmup: int = 10, iters: int = 100):
        for _ in range(warmup):
            _ = plan.execute(features, weights)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = plan.execute(features, weights)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters  # ms per iter

    def test_bandwidth_vs_default_backend(self):
        """Measure effective bandwidth for stencil and default backend."""
        # Dense block of ~1M active voxels. Large enough that per-launch overhead
        # is amortized and the kernels actually saturate some fraction of HBM.
        side = 96  # 96^3 ≈ 884k active voxels
        base = 4
        ii = torch.arange(side, device=self.DEVICE)
        grid_i, grid_j, grid_k = torch.meshgrid(ii, ii, ii, indexing="ij")
        coords = torch.stack(
            [grid_i.flatten(), grid_j.flatten(), grid_k.flatten()], dim=1
        ).to(torch.int32) + base
        grid = create_grid_from_coords(coords, self.DEVICE)
        dst_grid = grid.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)

        n_in = grid.total_voxels
        n_out = dst_grid.total_voxels
        features = JaggedTensor(torch.randn((n_in, 1), device=self.DEVICE, dtype=self.DTYPE))
        weights = torch.randn((1, 1, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)

        stencil_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid,
            target_grid=dst_grid,
            expert_config=STENCIL_CONFIG,
        )
        default_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid,
            target_grid=dst_grid,
        )

        stencil_ms = self._benchmark_plan(stencil_plan, features, weights)
        default_ms = self._benchmark_plan(default_plan, features, weights)

        useful_bytes = (n_in + n_out) * self.BYTES_PER_FLOAT32

        def bw_gbps(ms):
            return useful_bytes / (ms * 1e-3) / 1e9

        stencil_bw = bw_gbps(stencil_ms)
        default_bw = bw_gbps(default_ms)

        print()
        print(f"[bandwidth] n_in={n_in}, n_out={n_out}, useful_bytes={useful_bytes/1e6:.2f} MB")
        print(
            f"[bandwidth] stencil : {stencil_ms:.3f} ms/iter, "
            f"{stencil_bw:.2f} GB/s ({100 * stencil_bw / self.A6000_PEAK_GBPS:.1f}% of A6000 peak)"
        )
        print(
            f"[bandwidth] default : {default_ms:.3f} ms/iter, "
            f"{default_bw:.2f} GB/s ({100 * default_bw / self.A6000_PEAK_GBPS:.1f}% of A6000 peak)"
        )
        print(f"[bandwidth] speedup (default / stencil) = {default_ms / stencil_ms:.2f}x")

        # Sanity: both backends should at minimum produce non-trivial throughput.
        self.assertGreater(stencil_bw, 0.0)
        self.assertGreater(default_bw, 0.0)

    # -------------------------------------------------------------------------
    # Divergence specialization (in_c=3, out_c=1)
    # -------------------------------------------------------------------------

    def _make_divergence_weights(self, fx: float = 0.5, fy: float = 0.5, fz: float = 0.5):
        """Standard 3D divergence stencil: central differences along each axis.

        weights[0, 0, 0|2, 1, 1] = ∓fx   (∂Fx/∂x)
        weights[0, 1, 1, 0|2, 1] = ∓fy   (∂Fy/∂y)
        weights[0, 2, 1, 1, 0|2] = ∓fz   (∂Fz/∂z)
        """
        w = torch.zeros((1, 3, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)
        w[0, 0, 0, 1, 1] = -fx
        w[0, 0, 2, 1, 1] = +fx
        w[0, 1, 1, 0, 1] = -fy
        w[0, 1, 1, 2, 1] = +fy
        w[0, 2, 1, 1, 0] = -fz
        w[0, 2, 1, 1, 2] = +fz
        return w

    def test_divergence_matches_gather_scatter_classical(self):
        """Standard ±0.5 divergence weights — stencil path matches gather_scatter."""
        cluster = get_cluster_edge_aligned(self.KERNEL_SIZE, self.DEVICE)
        grid = create_grid_from_coords(cluster, self.DEVICE)
        dst_grid = grid.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        n = len(cluster)

        features = JaggedTensor(torch.randn((n, 3), device=self.DEVICE, dtype=self.DTYPE))
        weights = self._make_divergence_weights()

        stencil_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid, target_grid=dst_grid,
            channel_pairs=((3, 1),),
            expert_config=STENCIL_DIVERGENCE_CONFIG,
        )
        gs_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid, target_grid=dst_grid,
            channel_pairs=((3, 1),),
            expert_config=GS_CONFIG,
        )
        torch.testing.assert_close(
            stencil_plan.execute(features, weights).jdata,
            gs_plan.execute(features, weights).jdata,
            rtol=RTOL, atol=ATOL,
        )

    def test_divergence_matches_gather_scatter_random_on_stencil(self):
        """Random on-stencil values for the 6 divergence taps; many impulses."""
        impulse_coords, _ = generate_hermit_impulses_dense(
            num_candidates=self.NUM_CANDIDATES,
            volume_shape=self.VOLUME_SHAPE,
            kernel_size=self.KERNEL_SIZE,
            impulse_value=1,
            dtype=self.DTYPE,
            device=self.DEVICE,
        )
        grid = create_grid_from_coords(impulse_coords, self.DEVICE)
        dst_grid = grid.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        n = grid.total_voxels

        features = JaggedTensor(torch.randn((n, 3), device=self.DEVICE, dtype=self.DTYPE))
        weights = self._make_divergence_weights(
            fx=torch.randn((), device=self.DEVICE, dtype=self.DTYPE).item(),
            fy=torch.randn((), device=self.DEVICE, dtype=self.DTYPE).item(),
            fz=torch.randn((), device=self.DEVICE, dtype=self.DTYPE).item(),
        )

        stencil_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid, target_grid=dst_grid,
            channel_pairs=((3, 1),),
            expert_config=STENCIL_DIVERGENCE_CONFIG,
        )
        gs_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid, target_grid=dst_grid,
            channel_pairs=((3, 1),),
            expert_config=GS_CONFIG,
        )
        torch.testing.assert_close(
            stencil_plan.execute(features, weights).jdata,
            gs_plan.execute(features, weights).jdata,
            rtol=RTOL, atol=ATOL,
        )

    def test_divergence_rejects_nonzero_off_stencil_weights(self):
        """A fully random (1,3,3,3,3) weight tensor must be rejected at execute."""
        cluster = get_cluster_edge_aligned(self.KERNEL_SIZE, self.DEVICE)
        grid = create_grid_from_coords(cluster, self.DEVICE)
        n = len(cluster)

        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid,
            channel_pairs=((3, 1),),
            expert_config=STENCIL_DIVERGENCE_CONFIG,
        )
        features = JaggedTensor(torch.randn((n, 3), device=self.DEVICE, dtype=self.DTYPE))
        weights = torch.randn((1, 3, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)
        with self.assertRaises(ValueError) as cm:
            plan.execute(features, weights)
        self.assertIn("divergence", str(cm.exception).lower())

    def test_divergence_rejects_wrong_input_channels(self):
        """Passing a 1-channel input to the divergence path must error at plan time."""
        cluster = get_cluster_edge_aligned(self.KERNEL_SIZE, self.DEVICE)
        grid = create_grid_from_coords(cluster, self.DEVICE)
        with self.assertRaises(ValueError):
            ConvolutionPlan.from_grid_batch(
                kernel_size=self.KERNEL_SIZE, stride=1,
                source_grid=grid,
                channel_pairs=((1, 1),),  # divergence expects in=3
                expert_config=STENCIL_DIVERGENCE_CONFIG,
            )

    # -------------------------------------------------------------------------
    # Gradient specialization (in_c=1, out_c=3)
    # -------------------------------------------------------------------------

    def _make_gradient_weights(self, fx: float = 0.5, fy: float = 0.5, fz: float = 0.5):
        """Standard 3D gradient: central differences, one output channel per axis."""
        w = torch.zeros((3, 1, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)
        w[0, 0, 0, 1, 1] = -fx; w[0, 0, 2, 1, 1] = +fx  # ∂/∂x → out 0
        w[1, 0, 1, 0, 1] = -fy; w[1, 0, 1, 2, 1] = +fy  # ∂/∂y → out 1
        w[2, 0, 1, 1, 0] = -fz; w[2, 0, 1, 1, 2] = +fz  # ∂/∂z → out 2
        return w

    def test_gradient_matches_gather_scatter_classical(self):
        cluster = get_cluster_edge_aligned(self.KERNEL_SIZE, self.DEVICE)
        grid = create_grid_from_coords(cluster, self.DEVICE)
        dst_grid = grid.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        n = len(cluster)

        features = JaggedTensor(torch.randn((n, 1), device=self.DEVICE, dtype=self.DTYPE))
        weights = self._make_gradient_weights()

        stencil_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid, target_grid=dst_grid,
            channel_pairs=((1, 3),),
            expert_config=STENCIL_GRADIENT_CONFIG,
        )
        gs_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid, target_grid=dst_grid,
            channel_pairs=((1, 3),),
            expert_config=GS_CONFIG,
        )
        torch.testing.assert_close(
            stencil_plan.execute(features, weights).jdata,
            gs_plan.execute(features, weights).jdata,
            rtol=RTOL, atol=ATOL,
        )

    def test_gradient_matches_gather_scatter_many_impulses(self):
        impulse_coords, _ = generate_hermit_impulses_dense(
            num_candidates=self.NUM_CANDIDATES,
            volume_shape=self.VOLUME_SHAPE,
            kernel_size=self.KERNEL_SIZE,
            impulse_value=1,
            dtype=self.DTYPE,
            device=self.DEVICE,
        )
        grid = create_grid_from_coords(impulse_coords, self.DEVICE)
        dst_grid = grid.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        n = grid.total_voxels

        features = JaggedTensor(torch.randn((n, 1), device=self.DEVICE, dtype=self.DTYPE))
        weights = self._make_gradient_weights(
            fx=torch.randn((), device=self.DEVICE, dtype=self.DTYPE).item(),
            fy=torch.randn((), device=self.DEVICE, dtype=self.DTYPE).item(),
            fz=torch.randn((), device=self.DEVICE, dtype=self.DTYPE).item(),
        )

        stencil_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid, target_grid=dst_grid,
            channel_pairs=((1, 3),),
            expert_config=STENCIL_GRADIENT_CONFIG,
        )
        gs_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid, target_grid=dst_grid,
            channel_pairs=((1, 3),),
            expert_config=GS_CONFIG,
        )
        torch.testing.assert_close(
            stencil_plan.execute(features, weights).jdata,
            gs_plan.execute(features, weights).jdata,
            rtol=RTOL, atol=ATOL,
        )

    def test_gradient_rejects_nonzero_off_stencil_weights(self):
        cluster = get_cluster_edge_aligned(self.KERNEL_SIZE, self.DEVICE)
        grid = create_grid_from_coords(cluster, self.DEVICE)
        n = len(cluster)

        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid,
            channel_pairs=((1, 3),),
            expert_config=STENCIL_GRADIENT_CONFIG,
        )
        features = JaggedTensor(torch.randn((n, 1), device=self.DEVICE, dtype=self.DTYPE))
        weights = torch.randn((3, 1, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)
        with self.assertRaises(ValueError) as cm:
            plan.execute(features, weights)
        self.assertIn("gradient", str(cm.exception).lower())

    def test_gradient_rejects_wrong_output_channels(self):
        cluster = get_cluster_edge_aligned(self.KERNEL_SIZE, self.DEVICE)
        grid = create_grid_from_coords(cluster, self.DEVICE)
        with self.assertRaises(ValueError):
            ConvolutionPlan.from_grid_batch(
                kernel_size=self.KERNEL_SIZE, stride=1,
                source_grid=grid,
                channel_pairs=((1, 1),),  # gradient expects out=3
                expert_config=STENCIL_GRADIENT_CONFIG,
            )

    # -------------------------------------------------------------------------
    # MAC-grid Divergence / Gradient (forward / backward differences)
    # -------------------------------------------------------------------------

    def _make_mac_divergence_weights(self):
        """ConvSolve-style MAC divergence: forward difference along each axis."""
        w = torch.zeros((1, 3, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)
        # axis 0: u[i+1] - u[i]
        w[0, 0, 2, 1, 1] = +1.0; w[0, 0, 1, 1, 1] = -1.0
        # axis 1: v[j+1] - v[j]
        w[0, 1, 1, 2, 1] = +1.0; w[0, 1, 1, 1, 1] = -1.0
        # axis 2: w[k+1] - w[k]
        w[0, 2, 1, 1, 2] = +1.0; w[0, 2, 1, 1, 1] = -1.0
        return w

    def _make_mac_gradient_weights(self):
        """ConvSolve-style MAC gradient: backward difference per output channel."""
        w = torch.zeros((3, 1, 3, 3, 3), device=self.DEVICE, dtype=self.DTYPE)
        w[0, 0, 1, 1, 1] = +1.0; w[0, 0, 0, 1, 1] = -1.0   # ∂/∂x
        w[1, 0, 1, 1, 1] = +1.0; w[1, 0, 1, 0, 1] = -1.0   # ∂/∂y
        w[2, 0, 1, 1, 1] = +1.0; w[2, 0, 1, 1, 0] = -1.0   # ∂/∂z
        return w

    def test_mac_divergence_matches_gather_scatter(self):
        impulse_coords, _ = generate_hermit_impulses_dense(
            num_candidates=self.NUM_CANDIDATES,
            volume_shape=self.VOLUME_SHAPE,
            kernel_size=self.KERNEL_SIZE,
            impulse_value=1,
            dtype=self.DTYPE,
            device=self.DEVICE,
        )
        grid = create_grid_from_coords(impulse_coords, self.DEVICE)
        dst_grid = grid.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        n = grid.total_voxels

        features = JaggedTensor(torch.randn((n, 3), device=self.DEVICE, dtype=self.DTYPE))
        weights = self._make_mac_divergence_weights()

        stencil_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid, target_grid=dst_grid,
            channel_pairs=((3, 1),),
            expert_config={"backend": "stencil", "stencil": "mac-divergence"},
        )
        gs_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid, target_grid=dst_grid,
            channel_pairs=((3, 1),),
            expert_config=GS_CONFIG,
        )
        torch.testing.assert_close(
            stencil_plan.execute(features, weights).jdata,
            gs_plan.execute(features, weights).jdata,
            rtol=RTOL, atol=ATOL,
        )

    def test_mac_gradient_matches_gather_scatter(self):
        impulse_coords, _ = generate_hermit_impulses_dense(
            num_candidates=self.NUM_CANDIDATES,
            volume_shape=self.VOLUME_SHAPE,
            kernel_size=self.KERNEL_SIZE,
            impulse_value=1,
            dtype=self.DTYPE,
            device=self.DEVICE,
        )
        grid = create_grid_from_coords(impulse_coords, self.DEVICE)
        dst_grid = grid.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        n = grid.total_voxels

        features = JaggedTensor(torch.randn((n, 1), device=self.DEVICE, dtype=self.DTYPE))
        weights = self._make_mac_gradient_weights()

        stencil_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid, target_grid=dst_grid,
            channel_pairs=((1, 3),),
            expert_config={"backend": "stencil", "stencil": "mac-gradient"},
        )
        gs_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1,
            source_grid=grid, target_grid=dst_grid,
            channel_pairs=((1, 3),),
            expert_config=GS_CONFIG,
        )
        torch.testing.assert_close(
            stencil_plan.execute(features, weights).jdata,
            gs_plan.execute(features, weights).jdata,
            rtol=RTOL, atol=ATOL,
        )


if __name__ == "__main__":
    unittest.main()
