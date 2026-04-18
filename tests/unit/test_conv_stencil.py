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


if __name__ == "__main__":
    unittest.main()
