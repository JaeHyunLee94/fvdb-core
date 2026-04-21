# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Minimal driver for `ncu` (Nsight Compute) profiling of StencilConv.

Does warmup + exactly one timed forward pass for a single (topology,
stencil) pair so the profile output is clean:

    ncu --target-processes all \\
        --kernel-name regex:stencilConvKernel \\
        --launch-skip 0 --launch-count 1 \\
        --set full -o profile_out \\
        python tests/wip/benchmark/profile_stencil_conv.py \\
        --topology dense_180 --stencil dense27

The `--launch-skip` should be >= warmup iters (default 5) so `ncu`
captures the first *post-warmup* launch. `--launch-count 1` limits
capture to a single launch; `ncu` is slow under `--set full`.
"""

from __future__ import annotations

import argparse

import torch
from fvdb.convolution_plan import _StencilConvBackend
from fvdb.utils.tests.convolution_utils import create_grid_from_coords

from fvdb import ConvolutionPlan, JaggedTensor

DEVICE = torch.device("cuda", 0)
DTYPE = torch.float32
KERNEL_SIZE = (3, 3, 3)

LAPLACE7_ON_STENCIL = [
    (1, 1, 1),
    (0, 1, 1), (2, 1, 1),
    (1, 0, 1), (1, 2, 1),
    (1, 1, 0), (1, 1, 2),
]


def dense_block_coords(side: int, base: int = 4) -> torch.Tensor:
    ii = torch.arange(side, device=DEVICE, dtype=torch.int32)
    gi, gj, gk = torch.meshgrid(ii, ii, ii, indexing="ij")
    return torch.stack([gi.flatten(), gj.flatten(), gk.flatten()], dim=1) + base


def sphere_coords(radius: int, base: int | None = None) -> torch.Tensor:
    if base is None:
        base = radius + 2
    r = radius
    ii = torch.arange(-r, r + 1, device=DEVICE, dtype=torch.int32)
    gi, gj, gk = torch.meshgrid(ii, ii, ii, indexing="ij")
    r2 = torch.tensor(r * r, device=DEVICE, dtype=torch.int32)
    mask = (gi * gi + gj * gj + gk * gk) <= r2
    coords = torch.stack([gi[mask], gj[mask], gk[mask]], dim=1) + base
    return coords.contiguous()


TOPOLOGIES = {
    "dense_96":   lambda: dense_block_coords(96),
    "dense_180":  lambda: dense_block_coords(180),
    "sphere_r140": lambda: sphere_coords(140),
}

STENCIL_CONFIG = {
    "dense27":  {"backend": "stencil"},
    "laplace7": {"backend": "stencil", "stencil": "laplace7", "validate_weights": False},
}


def make_weights() -> torch.Tensor:
    # Zero off-stencil positions so the tensor is valid for both Dense27 and Laplace7.
    w = torch.zeros((1, 1, 3, 3, 3), device=DEVICE, dtype=DTYPE)
    for i, j, k in LAPLACE7_ON_STENCIL:
        w[0, 0, i, j, k] = torch.randn((), device=DEVICE, dtype=DTYPE)
    return w


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topology", required=True, choices=list(TOPOLOGIES.keys()))
    p.add_argument("--stencil", required=True, choices=list(STENCIL_CONFIG.keys()))
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    coords = TOPOLOGIES[args.topology]()
    grid = create_grid_from_coords(coords, DEVICE)
    dst_grid = grid.conv_grid(kernel_size=KERNEL_SIZE, stride=1)

    features = JaggedTensor(torch.randn((grid.total_voxels, 1), device=DEVICE, dtype=DTYPE))
    weights = make_weights()

    plan = ConvolutionPlan.from_grid_batch(
        kernel_size=KERNEL_SIZE,
        stride=1,
        source_grid=grid,
        target_grid=dst_grid,
        expert_config=STENCIL_CONFIG[args.stencil],
    )
    assert isinstance(plan._backend, _StencilConvBackend)

    print(
        f"topology={args.topology}  stencil={args.stencil}  "
        f"n_in={grid.total_voxels:,}  n_out={dst_grid.total_voxels:,}  "
        f"warmup={args.warmup}"
    )

    for _ in range(args.warmup):
        plan.execute(features, weights)
    torch.cuda.synchronize()

    # The one launch we want ncu to profile:
    plan.execute(features, weights)
    torch.cuda.synchronize()
    print("done (1 profiled launch)")


if __name__ == "__main__":
    main()
