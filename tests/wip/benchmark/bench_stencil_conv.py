# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Standalone benchmark for the StencilConv sparse convolution backend.

For each (topology, operator) pair, runs a forward pass under both the
specialized stencil backend and the default `gather_scatter` backend with
matching weights and channel counts, then prints a grouped table.

Operators benchmarked:
  dense27      : in=1, out=1, all 27 taps
  laplace7     : in=1, out=1, center + 6 face neighbors
  divergence   : in=3, out=1, central differences along each axis
  gradient     : in=1, out=3, central differences along each axis

Effective bandwidth counts only the "useful" data movement:
    bytes = (N_in * in_c + N_out * out_c) * sizeof(float32)
This is a lower bound on device traffic but is what's directly
comparable to HBM peak.

Usage:
    python tests/wip/benchmark/bench_stencil_conv.py
    python tests/wip/benchmark/bench_stencil_conv.py --warmup 20 --iters 100
    python tests/wip/benchmark/bench_stencil_conv.py --only sphere_r140
    python tests/wip/benchmark/bench_stencil_conv.py --ops divergence,gradient
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import torch
from fvdb.convolution_plan import _StencilConvBackend
from fvdb.utils.tests.convolution_utils import create_grid_from_coords

from fvdb import ConvolutionPlan, JaggedTensor

DEVICE = torch.device("cuda", 0)
DTYPE = torch.float32
KERNEL_SIZE = (3, 3, 3)
BYTES_PER_F32 = 4
A6000_PEAK_GBPS = 768.0


# =============================================================================
# Topology builders
# =============================================================================


def dense_block_coords(side: int, base: int = 4) -> torch.Tensor:
    """Solid cubic block of side^3 active voxels."""
    ii = torch.arange(side, device=DEVICE, dtype=torch.int32)
    gi, gj, gk = torch.meshgrid(ii, ii, ii, indexing="ij")
    return (
        torch.stack([gi.flatten(), gj.flatten(), gk.flatten()], dim=1) + base
    )


def sphere_coords(radius: int, base: int | None = None) -> torch.Tensor:
    """Solid ball of radius `radius` — ~(4/3)*pi*r^3 active voxels."""
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
    "sphere_r140": lambda: sphere_coords(140),  # ~11.5M voxels
}


# =============================================================================
# Operator specs
# =============================================================================


def _laplace7_on_stencil_weights() -> torch.Tensor:
    """7 random nonzero positions in a [1,1,3,3,3] tensor, others exactly zero."""
    pos = [
        (1, 1, 1),
        (0, 1, 1), (2, 1, 1),
        (1, 0, 1), (1, 2, 1),
        (1, 1, 0), (1, 1, 2),
    ]
    w = torch.zeros((1, 1, 3, 3, 3), device=DEVICE, dtype=DTYPE)
    for i, j, k in pos:
        w[0, 0, i, j, k] = torch.randn((), device=DEVICE, dtype=DTYPE)
    return w


def _dense27_weights() -> torch.Tensor:
    """27 random weights in a [1,1,3,3,3] tensor — Dense27 reads them all."""
    return torch.randn((1, 1, 3, 3, 3), device=DEVICE, dtype=DTYPE)


def _divergence_weights() -> torch.Tensor:
    """Central-difference divergence weights (with random magnitudes per axis)."""
    fx = torch.randn((), device=DEVICE, dtype=DTYPE).item()
    fy = torch.randn((), device=DEVICE, dtype=DTYPE).item()
    fz = torch.randn((), device=DEVICE, dtype=DTYPE).item()
    w = torch.zeros((1, 3, 3, 3, 3), device=DEVICE, dtype=DTYPE)
    w[0, 0, 0, 1, 1] = -fx; w[0, 0, 2, 1, 1] = +fx
    w[0, 1, 1, 0, 1] = -fy; w[0, 1, 1, 2, 1] = +fy
    w[0, 2, 1, 1, 0] = -fz; w[0, 2, 1, 1, 2] = +fz
    return w


def _gradient_weights() -> torch.Tensor:
    """Central-difference gradient weights (random magnitudes per axis)."""
    fx = torch.randn((), device=DEVICE, dtype=DTYPE).item()
    fy = torch.randn((), device=DEVICE, dtype=DTYPE).item()
    fz = torch.randn((), device=DEVICE, dtype=DTYPE).item()
    w = torch.zeros((3, 1, 3, 3, 3), device=DEVICE, dtype=DTYPE)
    w[0, 0, 0, 1, 1] = -fx; w[0, 0, 2, 1, 1] = +fx
    w[1, 0, 1, 0, 1] = -fy; w[1, 0, 1, 2, 1] = +fy
    w[2, 0, 1, 1, 0] = -fz; w[2, 0, 1, 1, 2] = +fz
    return w


@dataclass(frozen=True)
class Operator:
    name: str                          # e.g. "laplace7"
    stencil_name: str | None           # passed as expert_config["stencil"]; None for default-stencil (dense27)
    in_c: int
    out_c: int
    weights_fn: Callable[[], torch.Tensor]


OPERATORS: dict[str, Operator] = {
    "dense27":    Operator("dense27",    None,         1, 1, _dense27_weights),
    "laplace7":   Operator("laplace7",   "laplace7",   1, 1, _laplace7_on_stencil_weights),
    "divergence": Operator("divergence", "divergence", 3, 1, _divergence_weights),
    "gradient":   Operator("gradient",   "gradient",   1, 3, _gradient_weights),
}


# =============================================================================
# Setup + timing
# =============================================================================


def _stencil_config(op: Operator) -> dict:
    cfg: dict = {"backend": "stencil"}
    if op.stencil_name is not None:
        cfg["stencil"] = op.stencil_name
    # Skip the off-stencil-zero check during timing so we measure pure kernel time.
    cfg["validate_weights"] = False
    return cfg


def _gs_config() -> dict:
    return {"backend": "gather_scatter"}


def build(topology: str, op: Operator, backend_kind: str):
    coords = TOPOLOGIES[topology]()
    grid = create_grid_from_coords(coords, DEVICE)
    dst_grid = grid.conv_grid(kernel_size=KERNEL_SIZE, stride=1)

    features = JaggedTensor(
        torch.randn((grid.total_voxels, op.in_c), device=DEVICE, dtype=DTYPE)
    )
    weights = op.weights_fn()

    expert_config = _stencil_config(op) if backend_kind == "stencil" else _gs_config()
    plan = ConvolutionPlan.from_grid_batch(
        kernel_size=KERNEL_SIZE,
        stride=1,
        source_grid=grid,
        target_grid=dst_grid,
        channel_pairs=((op.in_c, op.out_c),),
        expert_config=expert_config,
    )
    if backend_kind == "stencil":
        assert isinstance(plan._backend, _StencilConvBackend)

    return plan, features, weights, grid.total_voxels, dst_grid.total_voxels


def time_plan(plan, features, weights, warmup: int, iters: int) -> float:
    """Return mean ms per iteration, using CUDA events."""
    for _ in range(warmup):
        plan.execute(features, weights)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        plan.execute(features, weights)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# =============================================================================
# Formatting
# =============================================================================


def format_results_table(results: list[dict]) -> str:
    by_key: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_key[(r["topology"], r["op"])][r["backend"]] = r

    col_widths = (14, 11, 9, 12, 12, 11, 11, 8, 10)
    sep = "-" * (sum(col_widths) + len(col_widths) - 1)
    header = (
        f"{'topology':<{col_widths[0]}} "
        f"{'operator':<{col_widths[1]}} "
        f"{'backend':<{col_widths[2]}} "
        f"{'n_in':>{col_widths[3]}} "
        f"{'n_out':>{col_widths[4]}} "
        f"{'mean (ms)':>{col_widths[5]}} "
        f"{'bw (GB/s)':>{col_widths[6]}} "
        f"{'% peak':>{col_widths[7]}} "
        f"{'speedup':>{col_widths[8]}}"
    )

    lines = [
        "",
        sep,
        f"StencilConv vs gather_scatter — {torch.cuda.get_device_name(0)} "
        f"(peak {A6000_PEAK_GBPS:.0f} GB/s)",
        "speedup = default_mean / stencil_mean  (per row pair)",
        sep,
        header,
        sep,
    ]

    # Iterate in stable topology × operator order.
    for topo in TOPOLOGIES.keys():
        for op_name in OPERATORS.keys():
            rows = by_key.get((topo, op_name))
            if not rows:
                continue
            d_row = rows.get("default")
            for backend in ("stencil", "default"):
                r = rows.get(backend)
                if r is None:
                    continue
                if backend == "stencil" and d_row is not None:
                    speedup_str = f"{d_row['mean_ms'] / r['mean_ms']:>{col_widths[8] - 1}.2f}x"
                else:
                    speedup_str = f"{'-':>{col_widths[8]}}"
                lines.append(
                    f"{r['topology']:<{col_widths[0]}} "
                    f"{r['op']:<{col_widths[1]}} "
                    f"{r['backend']:<{col_widths[2]}} "
                    f"{r['n_in']:>{col_widths[3]},d} "
                    f"{r['n_out']:>{col_widths[4]},d} "
                    f"{r['mean_ms']:>{col_widths[5]}.3f} "
                    f"{r['bw_gbps']:>{col_widths[6]}.2f} "
                    f"{r['pct_peak']:>{col_widths[7] - 1}.1f}% "
                    f"{speedup_str}"
                )
            lines.append("")

    lines.append(sep)
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (untimed)")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations per case")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated topology names to run (default: all)",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default=None,
        help="Comma-separated operator names to run (default: all). "
             f"Available: {','.join(OPERATORS.keys())}",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    topos = list(TOPOLOGIES.keys())
    if args.only:
        requested = {t.strip() for t in args.only.split(",") if t.strip()}
        unknown = requested - set(topos)
        if unknown:
            raise SystemExit(f"Unknown topology: {sorted(unknown)}. Known: {topos}")
        topos = [t for t in topos if t in requested]

    ops = list(OPERATORS.keys())
    if args.ops:
        requested = {t.strip() for t in args.ops.split(",") if t.strip()}
        unknown = requested - set(ops)
        if unknown:
            raise SystemExit(f"Unknown operator: {sorted(unknown)}. Known: {ops}")
        ops = [t for t in ops if t in requested]

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {args.warmup} iters   Timed: {args.iters} iters")
    print()

    results: list[dict] = []
    for topo in topos:
        for op_name in ops:
            op = OPERATORS[op_name]
            for backend_kind in ("stencil", "default"):
                tag = f"{topo}/{op_name}/{backend_kind}"
                print(f"  {tag:<40s} ...", end="", flush=True)
                plan, feat, w, n_in, n_out = build(topo, op, backend_kind)
                mean_ms = time_plan(plan, feat, w, warmup=args.warmup, iters=args.iters)
                useful_bytes = (n_in * op.in_c + n_out * op.out_c) * BYTES_PER_F32
                bw_gbps = useful_bytes / (mean_ms * 1e-3) / 1e9
                pct_peak = 100 * bw_gbps / A6000_PEAK_GBPS
                print(f"  {mean_ms:7.3f} ms   {bw_gbps:7.2f} GB/s")
                results.append(
                    dict(
                        topology=topo,
                        op=op_name,
                        backend=backend_kind,
                        n_in=int(n_in),
                        n_out=int(n_out),
                        mean_ms=mean_ms,
                        bw_gbps=bw_gbps,
                        pct_peak=pct_peak,
                    )
                )
                del plan, feat, w
                torch.cuda.empty_cache()

    print(format_results_table(results))


if __name__ == "__main__":
    main()
