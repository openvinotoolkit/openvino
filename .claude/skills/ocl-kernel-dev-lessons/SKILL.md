---
name: ocl-kernel-dev-lessons
description: Cross-kernel lessons from closing the sdpa_ocl vs sdpa_micro gap (24x to within 3%) on Intel Xe/Xe-HPG - bottleneck decision tree, effect-size ladder, design principles, correctness discipline, fixed-policy gating, tool/environment traps, and a start-a-new-kernel checklist. Use when starting or tuning any GPU OpenCL/DPAS kernel, planning a perf investigation, or reviewing a performance claim.
---

# General lessons for developing/optimizing a new GPU kernel

Details and evidence: `src/plugins/intel_gpu/docs/ocl_perf_guide/09-general-lessons-for-new-kernels.md`, case study chapter 06, tools chapter 07, asm chapter 08.

## Diagnosis order (for a large gap)
1. **Does the comparison do the same work** — kernel name/geometry, `usm_device`, the final-linked reference binary, whether it fell back to a tier (opt). (The first piece of the 24x gap was usm_host.)
2. **spill/scratch** — SPILL=, XMX active. A huge unroll + an extra accumulator (`MICRO_MATH=1`) is a common cause. Limiting the unroll alone went 126→51 ms.
3. **memory message shape** — in the ISA, lane gather (16 B)/word-mov pack vs block read. Separate the alignment/pitch gate conditions to recover block IO (head-72 9.28×, h128 V 51→28 ms).
4. **control flow** — remove dynamic goto/join with a full-tile guard (28→19 ms).
5. **live range and GRF/occupancy** — 256 GRF halves occupancy. Never adopt on spill 0 alone.
6. Only then tile/prefetch/SLM tuning. (Textbook tuning was within the last 10%.)
7. If only short lengths are slow, it is a wave/prologue/pre-pass fixed cost → tiny geometry per length.
   If the gap **grows** with length, first check whether the per-call kernel ratio is constant and only the kernel's share grows (attention is quadratic): qwen3-8b 1.4x at every length became +2%..+24% e2e (`docs/ocl_perf_guide/01` §6.6).
   If the per-call **ratio** drifts with size, check the clock actually reached (PL2 overrides the B70 pin in long runs) and re-sweep at a fixed clock below the power limit; a remaining knee = a working set crossing a cache size (chapter 06 §6.7).
8. If IGC sinks loads and latency is exposed, go native (vISA) — `xe-visa-inline-asm`.
9. Utilisation equal to the reference but slower: compare **per-type instruction totals** (VTune overview: XMX/ALU) — +35% XMX instructions at equal activity was redundant work (the causal triangle of the diagonal tile), not scheduling (chapter 06 §6.8).

## Design principles
- Make the producer/consumer tile bound one value (`min(WGK, causal_k-k0)`); test tail/partial separately.
- Put a WG barrier wherever SLM ownership changes (alias race).
- Reducing branches is not the same as removing tail code: clamped whole-SV (always run fully + mask) can be faster, but KQ skip gives NaN under some lowerings.
- Separate settings for small shapes (Q8/K64/128 GRF). Instead of one fixed tile, a `(head, Qheads, KVheads, max_len)` selection policy; for multiple sequences, select by the **maximum length**.
- Head count and head size are independent axes — an improvement on one shape hides a defect on another.
- Mirror the reference's **workgroup** tile per head size (K/V re-read = seq / wgTQ), then pipeline the loop like it (branchless exp, prefetch inside a split barrier, last-tile mask). Check every head size a model mixes (gemma-4: 256 + 512); one left on the old tile made the model 2.2x slower.
- A pre-pass (external layout rearrangement) is for diagnosis; if replacing it is the goal, solve it inside the kernel (fixed cost 44–110 µs).
- Measure HW facts first (DPAS 1 in-flight, L1 prefetch hurts, SLM 47.7 KB → occupancy halved, B70 L3 line 256 B).
- Count a prefetch in cache-line requests: one request per line, into the level that misses, only where the working set exceeds the cache (64 B stride +17%, 256 B -0.5% at 8K / -11% at 32K; chapter 02 §7.6).
- A faster kernel exposes more latency: after overtaking the reference, re-check the large-size end.
- Layout does not fit the read? Permute the **contraction axis** instead of the data: a 32 B-row d-major page block-read through a wider-pitch surface + 32b transpose reached token-major speed without changing the cache contract (chapter 09 §9.3 #15). Probe the builtin's lane layout first.
- Write each DPAS operand row once on every path (a post-read `if` fix-up = phi copies); keep code the configuration never needs out of the hot loop (the f16 Kc branch: 5-14%); a skip is a branch every iteration pays (the causal skip won at d128+ and lost 6.8% at d64 4K / 4.7% in MIXED) — measure per tiling and stage (chapter 09 §9.3 #16-18).
- "Fewer messages, more ALU" is regime-dependent (512 B Q_slm read: 8 movs per read; won when Send/memory-bound, lost in cache-resident short prompts and small windows), bits do not add up, and a sign can flip with the kernel variant (sink). A per-tiling default reaches every variant: measure each, gate on compile-time features (`desc->sliding_window`) when the mechanism supports it (chapter 09 §9.3 #14).

- Private arrays take compile-time indices only: IGC 2.41 left a uniformly indexed one in scratch (KV-cache writer `TPM=576`, PREFILL 1.4x master -> 0.77x after the fix). After any driver upgrade, re-read `SPILL=`/`TPM=` of every kernel and re-measure the baseline.
- Threads writing the same cache lines should not write them in the same order: 16 partitions storing into the same 128 B rows queued (5.75 us); a per-partition start row fixed it (5.03) where co-location in one workgroup had not. Removing one cost (scratch reloads) can expose this kind of second one: sweep every shape after a fix.

## Accuracy
Accuracy before timing. NaN poison + CPU double on **all rows** (sampled rows are not exhaustive), fixed tolerance. Math-invariant changes must be bit-identical to the previous variant. Sharp inputs (smooth N(0,0.1) hides Q/K·softmax errors). Check for LCG power-of-two degenerate data. Boundary scan (1,16,31–33,63–65,127–129,255–257,511–513,1025,1033,2048,4096,8192; odd depth; padding NaN). **Never revive a candidate on a single-length PASS** (seq256 PASS/257 NaN case). Check the producer's bytes, not only the consumer's output: two broken KV-cache writers passed every paged-attention gtest row; a per-row cache dump compared byte for byte caught both.

## Experiment management
1. **Freeze** the fixed policy and source SHA256 **before measuring**, stop at the first FAIL — no cherry-picking.
2. One thing at a time, bidirectional controls, no-op/branch-only controls (confirm the final ISA SHA256 is identical).
3. Independent repeats (seed, reverse order, both cadences: queued32 / wait_each gap200). Never adopt a small improvement (≤2%) before a repeat.
4. Never pass a threshold by rounding. Report both the arm median and the same-round paired ratio.
5. Leave failed/excluded runs and their reasons, and the do-not-retry list (dead ends), in the HANDOFF. Also record overlap incidents and do a clean repeat.
6. Distinguish evidence layers: standalone harness PASS ≠ product dispatch ≠ product gtest ≠ e2e ≠ regression gate (S6/SG16/B70). Write the coverage holes (verified head/count/padding/feature) next to the PASS table.
7. Metrics (spill, occupancy, XMX active, bandwidth, instruction count, IPC) are clues; the verdict is a matched device-time A/B.
8. GPU jobs run serially. Instrumentation (phase timer/GTPin) cannot be used for acceptance.

## New-kernel start checklist
Define the call (operation/dtype/shape/layout/mask/device/driver/cadence) → fix the reference binary → confirm it does the same work → length/shape scan table → runtime evidence (device time+geometry, SPILL/TPM, GRF/SLM) → one hypothesis·prediction → one change·control·accuracy→performance → widen the gate (short/long/boundary/head-count/cadence) → record results·failures·holes·commands → product integration is a separate step (census, gtest, byte-identity on other architectures, e2e).
