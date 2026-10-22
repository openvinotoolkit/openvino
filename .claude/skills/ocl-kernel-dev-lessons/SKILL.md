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
8. If IGC sinks loads and latency is exposed, go native (vISA) — `xe-visa-inline-asm`.

## Design principles
- Make the producer/consumer tile bound one value (`min(WGK, causal_k-k0)`); test tail/partial separately.
- Put a WG barrier wherever SLM ownership changes (alias race).
- Reducing branches is not the same as removing tail code: clamped whole-SV (always run fully + mask) can be faster, but KQ skip gives NaN under some lowerings.
- Separate settings for small shapes (Q8/K64/128 GRF). Instead of one fixed tile, a `(head, Qheads, KVheads, max_len)` selection policy; for multiple sequences, select by the **maximum length**.
- Head count and head size are independent axes — an improvement on one shape hides a defect on another.
- A pre-pass (external layout rearrangement) is for diagnosis; if replacing it is the goal, solve it inside the kernel (fixed cost 44–110 µs).
- Measure HW facts first (DPAS 1 in-flight, L1 prefetch hurts, SLM 47.7 KB → occupancy halved).

## Accuracy
Accuracy before timing. NaN poison + CPU double on **all rows** (sampled rows are not exhaustive), fixed tolerance. Math-invariant changes must be bit-identical to the previous variant. Sharp inputs (smooth N(0,0.1) hides Q/K·softmax errors). Check for LCG power-of-two degenerate data. Boundary scan (1,16,31–33,63–65,127–129,255–257,511–513,1025,1033,2048,4096,8192; odd depth; padding NaN). **Never revive a candidate on a single-length PASS** (seq256 PASS/257 NaN case).

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
