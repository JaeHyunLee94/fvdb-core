# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Standalone benchmark for the StencilConv sparse convolution backend.

Runs a forward pass under both the `stencil` backend and the default
(`gather_scatter`) backend across a few large topologies and prints a
grouped table with timing, effective bandwidth, and % of A6000 peak.

Effective bandwidth counts only the "useful" data movement:
    bytes = (N_in + N_out) * sizeof(float32)
This is a lower bound on device traffic (halo gathers re-read neighbors,
NanoVDB index lookups add more traffic), but it is the metric most
directly comparable to HBM peak.

Usage:
    python tests/wip/benchmark/bench_stencil_conv.py
    python tests/wip/benchmark/bench_stencil_conv.py --warmup 20 --iters 100
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

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
    "dense_96": lambda: dense_block_coords(96),
    "dense_180": lambda: dense_block_coords(180),
    "sphere_r140": lambda: sphere_coords(140),  # ~11.5M voxels
}

BACKENDS = {
    "stencil": {"backend": "stencil"},
    "default": {"backend": "gather_scatter"},
}


# =============================================================================
# Setup + timing
# =============================================================================


def build(topology: str, backend: str):
    coords = TOPOLOGIES[topology]()
    grid = create_grid_from_coords(coords, DEVICE)
    dst_grid = grid.conv_grid(kernel_size=KERNEL_SIZE, stride=1)

    features = JaggedTensor(
        torch.randn((grid.total_voxels, 1), device=DEVICE, dtype=DTYPE)
    )
    weights = torch.randn((1, 1, 3, 3, 3), device=DEVICE, dtype=DTYPE)

    plan = ConvolutionPlan.from_grid_batch(
        kernel_size=KERNEL_SIZE,
        stride=1,
        source_grid=grid,
        target_grid=dst_grid,
        expert_config=BACKENDS[backend],
    )
    if backend == "stencil":
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
    by_topo: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_topo[r["topology"]][r["backend"]] = r

    col_widths = (14, 9, 12, 12, 11, 11, 8, 10)
    sep = "-" * (sum(col_widths) + len(col_widths) - 1)
    header = (
        f"{'topology':<{col_widths[0]}} "
        f"{'backend':<{col_widths[1]}} "
        f"{'n_in':>{col_widths[2]}} "
        f"{'n_out':>{col_widths[3]}} "
        f"{'mean (ms)':>{col_widths[4]}} "
        f"{'bw (GB/s)':>{col_widths[5]}} "
        f"{'% peak':>{col_widths[6]}} "
        f"{'speedup':>{col_widths[7]}}"
    )

    lines = [
        "",
        sep,
        f"StencilConv vs default backend  —  {torch.cuda.get_device_name(0)}  "
        f"(peak {A6000_PEAK_GBPS:.0f} GB/s)",
        "speedup = default_mean / stencil_mean",
        sep,
        header,
        sep,
    ]

    for topo in TOPOLOGIES.keys():
        rows = by_topo.get(topo)
        if not rows:
            continue
        s_row = rows.get("stencil")
        d_row = rows.get("default")
        speedup = (d_row["mean_ms"] / s_row["mean_ms"]) if (s_row and d_row) else None

        for backend in ("stencil", "default"):
            r = rows.get(backend)
            if r is None:
                continue
            speedup_str = (
                f"{speedup:>{col_widths[7] - 1}.2f}x"
                if backend == "stencil" and speedup is not None
                else f"{'-':>{col_widths[7]}}"
            )
            lines.append(
                f"{r['topology']:<{col_widths[0]}} "
                f"{r['backend']:<{col_widths[1]}} "
                f"{r['n_in']:>{col_widths[2]},d} "
                f"{r['n_out']:>{col_widths[3]},d} "
                f"{r['mean_ms']:>{col_widths[4]}.3f} "
                f"{r['bw_gbps']:>{col_widths[5]}.2f} "
                f"{r['pct_peak']:>{col_widths[6] - 1}.1f}% "
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

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {args.warmup} iters   Timed: {args.iters} iters")
    print()

    results: list[dict] = []
    for topo in topos:
        for backend in BACKENDS.keys():
            print(f"  building {topo}/{backend} ...", end="", flush=True)
            plan, feat, w, n_in, n_out = build(topo, backend)
            print(f" n_in={n_in:,}, n_out={n_out:,}", end="", flush=True)
            mean_ms = time_plan(plan, feat, w, warmup=args.warmup, iters=args.iters)
            useful_bytes = (n_in + n_out) * BYTES_PER_F32
            bw_gbps = useful_bytes / (mean_ms * 1e-3) / 1e9
            pct_peak = 100 * bw_gbps / A6000_PEAK_GBPS
            print(f"  ->  {mean_ms:.3f} ms   {bw_gbps:.2f} GB/s")
            results.append(
                dict(
                    topology=topo,
                    backend=backend,
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
