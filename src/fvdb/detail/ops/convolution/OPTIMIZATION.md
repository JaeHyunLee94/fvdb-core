# StencilConv — Optimization Strategies

This document tracks optimization strategies for the StencilConv sparse
convolution kernel (`StencilConv.cu`). It is meant as a living notebook —
the two big sections are:

- [**What we already tried**](#what-we-already-tried) — every change we
  evaluated, with measured performance impact per change. Includes work
  that landed on `main` and work that was tried and reverted.
- [**What we can potentially try in the future**](#what-we-can-potentially-try-in-the-future)
  — candidate optimizations that have not yet been implemented, ranked by
  expected payoff vs. effort.

---

## 0. Kernel shape and baseline

`StencilConv.cu` runs a forward-only sparse 3D convolution:

- One CUDA block per target leaf (8³ = 512 output voxels → 512 threads/CTA).
- A 10×10×10 halo of source-grid feature values is staged into shared memory.
- Each thread performs one multiply–accumulate over a compile-time stencil
  descriptor's tap set and writes one output.

Weights tensor stays `[1,1,3,3,3]`; the stencil descriptor (`Dense27Stencil`
or `Laplace3DStencil`) picks which of the 27 positions are actually read.

### Current baseline (HEAD: commit `9eb4272`)

Hardware: RTX A6000 (SM86, 1536 threads/SM, peak HBM 768 GB/s).

| topology | Dense27 | Laplace7 | Laplace7 % of peak | vs `gather_scatter` |
|---|---|---|---|---|
| dense_96 (0.88 M voxels) | 0.083 ms, 88 GB/s | 0.066 ms, 111 GB/s | 14.4 % | 32.5× |
| dense_180 (5.83 M) | 0.454 ms, 105 GB/s | 0.360 ms, 132 GB/s | 17.2 % | 41.3× |
| sphere_r140 (11.5 M) | 0.884 ms, 106 GB/s | 0.690 ms, 135 GB/s | 17.6 % | 60.4× |

Reproduce with:

```
python tests/wip/benchmark/bench_stencil_conv.py
```

Kernel-only timings via nsys (no root perms required):

```
nsys profile --trace=cuda --stats=true --force-overwrite true -o /tmp/s.out \
    python tests/wip/benchmark/profile_stencil_conv.py --topology dense_180 --stencil dense27
```

Inspect register / smem usage without any perms:

```
cuobjdump --dump-resource-usage build/.../libfvdb.so | grep -A2 stencilConvKernel
```

`ncu` counter-based profiling currently requires admin:
`NVreg_RestrictProfilingToAdminUsers=0`.

### Where time is spent (inferred from the Dense27 vs Laplace7 delta)

Phase 1 (halo load + target tree walk + `__syncthreads` + output store) is
~68 % of Dense27 kernel time. Phase 2 (accumulation) is ~32 %. Occupancy is
thread-count-limited at 3 CTAs / SM (1536 threads = 75 % of SM86 warp slots);
smem (4 KB × 3 = 12 KB) and registers (40 × 1536 = 61 k of 64 k) are both
under the limit, so adding a little smem or a handful of registers is free.

---

## What we already tried

Each row below is one change we evaluated, in chronological order. "Status"
is `merged` if it lives on the current tip (`9eb4272`), `reverted` if we
tried it and rolled back for staging. Both the merged and reverted work is
worth reading — the reverted change has measured data and is ready to
cherry-pick if we decide to build on it.

### 2.1 Shared memory holds halo **values**, not indices
- **Commit:** `b6b5d94`
- **Status:** merged
- **Change:** `__shared__ int64_t haloIndices[10][10][10]` → `__shared__ float
  haloValues[10][10][10]`. The per-tap `inputFeatures[raw-1]` global load and
  the `if (inIdx >= 0)` guard move out of the 27-tap hot loop; inactive slots
  are zero-filled once at halo-load time so the accumulation is branch-free.
- **Expected:** major — hot-loop turns from "27 guarded scattered global loads"
  into "27 shared-memory FMAs".
- **Measured (end-to-end wall clock):**
  - dense_96 : 0.099 → 0.085 ms (−14 %, 74 → 86 GB/s)
  - dense_180 : 0.544 → 0.468 ms (−14 %, 87 → 102 GB/s)
  - sphere_r140: 1.046 → 0.916 ms (−12 %, 89 → 102 GB/s)
- **Notes:** also halved shared-memory footprint (8 KB → 4 KB), but that
  didn't unlock more occupancy because the thread-count cap binds first.

### 2.2 Bit-op leaf-local index decoding
- **Commit:** `b6b5d94` (same commit as 2.1)
- **Status:** merged
- **Change:** `li = tid / 64; lj = (tid / 8) % 8; lk = tid % 8` → `li = (tid >>
  6) & 0x7; lj = (tid >> 3) & 0x7; lk = tid & 0x7`.
- **Expected:** ~0 % — `nvcc` already strength-reduces `/` and `%` on
  power-of-two constants. Change is for source clarity.
- **Measured:** flat within measurement noise (< 1 %). Confirmed that the
  compiler was already doing the substitution at PTX level.

### 2.3 Replace CUDA-side `ReadAccessor` with direct `tree().getValue()`
- **Commit:** `a0217d9`
- **Status:** merged
- **Change:** Drop the per-thread `auto srcAcc = sourceGrid->getAccessor();`
  stack-allocated cache for both source and target lookups; call the tree
  directly. Each thread only does 1–2 lookups, so the accessor's cached leaf
  pointer never pays back its init cost.
- **Expected:** small — saves some stack bytes and a cache-init.
- **Measured (wall clock):**
  - dense_96 : 0.085 → 0.083 ms (−2.4 %)
  - dense_180 : 0.466 → 0.457 ms (−1.9 %)
  - sphere_r140: flat
- **Notes:** modest, but free (pure deletion).

### 2.4 Compile-time sparse stencil (Dense27 / Laplace7)
- **Commit:** `2cc1c69`
- **Status:** merged
- **Change:** Kernel templated on a `StencilDescriptor` type whose nested
  `Taps` member is a `std::tuple` of `StencilPoint<DI,DJ,DK>`. The triple
  (di,dj,dk) accumulation loop becomes a compile-time fold over
  `StencilT::Taps`. Two explicit specializations ship: `Dense27Stencil` (27
  taps, default) and `Laplace3DStencil` (center + 6 face neighbours).
  Python gates via `expert_config={"backend":"stencil","stencil":"laplace7"}`;
  off-stencil weight positions are validated at execute time (cached
  per-device index; escape hatch via `validate_weights=False`).
- **Expected:** ~(27 − 7) / 27 of phase-2 cost, i.e. ~20 % end-to-end.
- **Measured (Laplace7 vs Dense27, same weights):**
  - dense_96 : 0.083 → 0.066 ms (−20 %, 88 → 111 GB/s)
  - dense_180 : 0.454 → 0.360 ms (−21 %, 105 → 132 GB/s)
  - sphere_r140: 0.884 → 0.690 ms (−22 %, 106 → 135 GB/s)
- **Notes:** matches the "27 → 7 FMAs while halo load is unchanged" model.
  Registers dropped from 40 → 32 on the Laplace7 variant but that didn't
  buy extra occupancy (thread-count cap still binds).

### 2.5 Leaf-direct halo load (tried, reverted)
- **Commit (applied):** `0986fa4`
- **Commit (revert):** `9eb4272`
- **Status:** reverted — kept in history, re-apply with
  `git cherry-pick 0986fa4`.
- **Change:** Replace the per-halo-slot `srcTree.getValue(ijk)` root-to-leaf
  walk with a two-phase scheme:
  - **Phase 0 (27 threads):** probe the 3×3×3 neighbourhood of source leaves
    once, cache pointers in shared memory.
  - **Phase 1 (reworked):** each halo slot identifies its owning leaf via
    `(gx >> 3) - (Lx >> 3) + 1` bit math and calls `leaf->getValue(localOff)`
    — an O(1) popcount on the leaf's `mValueMask`, no tree walk.
  - **Target lookup:** `outLeaf` is already in hand, so the per-thread
    `targetGrid->tree().getValue(outIJK)` becomes
    `outLeaf.getValue((li<<6)|(lj<<3)|lk)`.
  - Tree traversals per CTA drop from ~1001 to 27.
- **Expected:** ~30 % if phase 1 is ~68 % of the kernel and leaf-direct halves
  it.
- **Measured (end-to-end wall clock):**
  - dense_96 : 0.083 → 0.073 ms (−12 %, 88 → 100 GB/s) / 0.066 → 0.057 ms
    Laplace7 (−14 %)
  - dense_180 : 0.454 → 0.400 ms (−12 %, 105 → 119 GB/s) / 0.360 → 0.311 ms
    Laplace7 (−14 %)
  - sphere_r140: 0.884 → 0.759 ms (−14 %, 106 → 123 GB/s) / 0.690 → 0.590 ms
    Laplace7 (−14 %)
- **Measured (nsys kernel-only, dense_180):**
  - Dense27 : 387 → 332 µs (−14 %)
  - Laplace7: 295 → 243 µs (−18 %)
  - The roughly-constant ~55 µs absolute saving across both specializations
    confirms phase 1 was the target.
- **Resource deltas:**
  - Smem: 4000 → 4216 B (+27 × 8 B leaf-pointer cache). Still well under the
    occupancy limit.
  - Regs: Dense27 40 → 40 (unchanged). Laplace7 32 → 40. Thread-count cap
    still binds on SM86, so neither specialization lost a CTA/SM.
- **Why reverted:** rolled back for staging — the change was correct
  (all 21 cross-backend tests green at `rtol=atol=1e-5`) but we chose to
  build up the experiment list first before layering more optimizations
  on top. `git cherry-pick 0986fa4` re-applies it cleanly.

---

## What we can potentially try in the future

Not-yet-implemented ideas, ranked by expected payoff / effort. The top few
are low-risk edits; the later items are bigger rewrites with bigger ceilings.

### 3.1 Bank-conflict padding for `haloValues`
- **Expected win:** 2–5 % end-to-end.
- **Effort:** 1 line of code.
- **Change:** `__shared__ float haloValues[10][10][10]` →
  `[10][10][10 + 1]`. The innermost-dim stride of 10 floats (40 B) maps
  imperfectly onto the 32 smem banks on SM86 — adjacent threads in phase 2
  stepping over `lk` can collide. Padding to 11 floats shifts the stride.
- **Risk:** none. Output layout is unchanged; only the smem access stride.

### 3.2 `__ldg` on the global feature fetch
- **Expected win:** 5–10 % end-to-end.
- **Effort:** 1 line of code.
- **Change:** `inputFeatures[raw - 1]` →
  `__ldg(inputFeatures + raw - 1)`. Routes the scattered halo-feature read
  through the read-only data cache (texture unit) instead of competing with
  L1 for the halo writes. `inputFeatures` is already tagged `__restrict__`.
- **Risk:** none.

### 3.3 Re-apply leaf-direct halo load (cherry-pick `0986fa4`)
- **Expected win:** 12–18 % (measured — see section 2.5).
- **Effort:** `git cherry-pick 0986fa4`; possibly rebase.
- **Change:** see 2.5.
- **Risk:** same as before; test suite covers it.

### 3.4 `cp.async` (`cuda::memcpy_async`) halo load
- **Expected win:** 10–20 %.
- **Effort:** rewrite of phase 1, ~30–50 lines.
- **Change:** issue the 1000 halo feature fetches via SM80+ `cp.async` so the
  post-phase-1 `__syncthreads()` overlaps device → smem transfer with
  everything else in the CTA (phase-0 leaf probe, thread decode, output
  lookup). `cp.async` has native indirect-index variants; each slot is still
  a one-element transfer so this mostly helps via latency hiding, not
  bandwidth. Compose with 3.3 (leaf-direct) for a cleaner structure.
- **Risk:** moderate — easy to miss a barrier. Use `cuda::pipeline` +
  `pipeline.wait_prior<0>()` as in the CUDA samples.

### 3.5 Warp-cooperative, leaf-contiguous halo load
- **Expected win:** 20–30 % (highest ceiling of the untried options).
- **Effort:** significant rewrite, ~80–120 lines.
- **Change:** today halo slot `s` is handled by thread `s` regardless of
  which source leaf it lives in, so a single warp's 32 threads can scatter
  reads across up to 8 different source leaves. Restructure phase 1 so each
  warp owns exactly one source leaf and streams its active-voxel feature
  values as a contiguous chunk (NanoVDB stores them sequentially in a
  `ValueOnIndex` leaf), then writes them into the appropriate halo
  positions. Feature reads go from scattered to fully coalesced.
- **Risk:** high — mask-based popcount gymnastics. Lean on the 21
  cross-backend tests.

### 3.6 Thread-coarsening for Laplace7
- **Expected win:** 20–30 % (Laplace7 only, doesn't touch Dense27).
- **Effort:** new Laplace7-specialized kernel variant.
- **Change:** each thread handles a 1×1×2 or 2×1×1 tile of output voxels.
  Adjacent Laplacian outputs share 5 of 7 taps (4 in-plane neighbours + the
  center, when stepping by 1 along one axis), so reuse halo reads in
  registers. Halves smem-read traffic per output. Will likely cost
  registers; may drop occupancy from 3 → 2 CTAs / SM, so needs careful
  measurement.
- **Risk:** moderate. Correctness unaffected if the tile mapping is right.

### 3.7 Phase-strip instrumentation variants
- **Expected win:** none (it's instrumentation).
- **Effort:** small, but must be gated.
- **Change:** compile-time flag `STENCIL_PROFILE_MODE` that, when set, lets
  the kernel skip phase 1 (zero-fill halo) or skip phase 2 (write
  `haloValues[center]`) for phase-cost attribution when `ncu` counters are
  unavailable. **Must default to 0** and never ship as the default build —
  earlier we learned the hard way that intentionally-breaking kernel
  variants in production code is flagged as destructive.
- **Risk:** negligible with the default-off gate.

---

## 4. Explicitly rejected / low-priority

Attractive on paper but the profile says they won't move the needle here:

- **Weights in `__constant__` memory.** 27 floats read identically by every
  thread; they live in L1 after the first read. Saves < 1 %.
- **Persistent-threads / launch-overhead amortization.** Launch overhead is
  ~5 µs (nsys); kernel time is 100–900 µs. Not a bottleneck.
- **Smaller CTA sizes.** 128 or 256 threads/CTA would allow more CTAs per
  SM, but the algorithm is built around "one thread per leaf voxel" (512).
  Going smaller forces a redesign; going larger is blocked by the 1536
  threads/SM cap.
- **Register reduction for Dense27.** Already at 40 regs/thread with no
  spill; the thread-count limit (3 CTAs/SM) binds before the register limit
  (3.2 CTAs/SM). Cutting regs gives 0 % occupancy gain.

---

## 5. How to evaluate a candidate

1. Implement on a feature branch; keep changes narrowly scoped.
2. Rebuild: `./build.sh` (incremental — one TU recompiles).
3. **Correctness first:** `python -m pytest tests/unit/test_conv_stencil.py -v`.
   All 21 cross-backend tests must stay green at `rtol=atol=1e-5`. These
   cover single impulse, dense cluster, 1000 hermit impulses, sparse
   Laplacian, disjoint src/tgt grids, hollow shell, leaf-boundary spanning,
   random dropout, thin diagonal curve, Laplace7 bit-identical with Dense27,
   Laplace7 weight-validation rejections, etc.
4. **Perf:** `python tests/wip/benchmark/bench_stencil_conv.py` on an idle
   GPU (check `nvidia-smi` — util 0 %, temp < 50 °C). Run at least twice;
   contention on a shared GPU can double wall time.
5. Inspect kernel-only timings with `nsys` if the Python-level numbers look
   too noisy. Inspect register/smem with `cuobjdump` if you're worried about
   occupancy.
6. If correctness passes and perf improves, commit with a message that
   includes the before → after ms/GB/s table for all three topologies,
   consistent with the entries in section 2 above.

---

## 6. Suggested next sequence

1. Bundle **3.1 + 3.2** (bank-conflict padding + `__ldg`) into one commit.
   Both are one-line edits; together probably ~10 %. Decouples the small
   wins from larger rewrites.
2. Re-apply **3.3** (leaf-direct halo). Measured −14 % wall clock, already
   in history as `0986fa4`.
3. Layer **3.4** (`cp.async`) on top of 3.3.
4. Choose between **3.5** and **3.6** based on what the profile says after
   3.4.

Past this point we are likely within a factor of 2 of the HBM ceiling for
the current problem shape; further gains probably require changing the CTA
decomposition (multi-leaf CTAs sharing halo loads across adjacent output
leaves) or fusing conv stages at a higher level.
