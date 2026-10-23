---
name: ocl-kernel-ab-methodology
description: Playbook for proving and measuring changes to high-performance OpenCL/DPAS GPU kernels (sdpa_ocl, paged attention) in OpenVINO intel_gpu - offline A/B (L0/L1/L2/pset/xab/built), attribution with both controls, oracle design, benchmark discipline. Use when A/B testing, bisecting a kernel failure, claiming a speedup, or reviewing kernel results.
---

# OCL kernel A/B·attribution·benchmark methodology (playbook)

In-depth explanations and tables: `src/plugins/intel_gpu/docs/ocl_perf_guide/05-methodology-and-pitfalls.md` (hereafter "chapter 05"). Kernel techniques are in chapters 01~04.
Repository rules (`AGENTS.md`): numerical correctness first, no performance claims without measurement, state the methodology, distinguish measured vs assumed, no scope expansion/opportunistic cleanup.
For investigating comparison-arm errors and unexplained gaps and for tool selection, see [`ocl-kernel-performance-investigation`](../ocl-kernel-performance-investigation/SKILL.md) and [chapter 06](../../../src/plugins/intel_gpu/docs/ocl_perf_guide/06-performance-gap-investigation.md).

## 0. Work rules to follow first
- **Builds, gtests, ab (full corpus), dumps, benchmarks, `-fsyntax-only`, `built` and `l0` are run by the user.** The agent presents exact command blocks + predicted values (the only exception is when direct execution is explicitly permitted, and even then re-confirm every session).
- **The user commits.** Only propose a one-line message. Rationale/history go in code comments and `docs/sdpa_ocl.md`.
- Get plan approval before changing code. Do not fix out-of-scope findings; record them in `docs/sdpa_ocl.md` "Known issues".
- Before editing a bash script the user may be running, run `ps -eo pid,etime,args | grep <script>` (if the length changes, bash executes the tail of the file as commands). If unavoidable, keep the byte length the same.
- For long tasks, at the end of the session update memory + write a Korean resume prompt. Poll in `sleep 240` steps.

## 1. Kind of change -> choice of proof
| Change | Proof |
|---|---|
| `.cl` behaviour-preserving (refactor/helper extraction) | `sdpa_ocl_ab.py l0`, `ab --levels l1,l1p,l2 --l2-from-l1` with A=all over the whole corpus. Helpers **must be `always_inline`** (plain `inline` changes the whole kernel's ISA) |
| `.cl` bug fix | In `ab`, **only the target configs** may leave A |
| Host-only | New `dump` -> `corpus` -> `pset <baseline> <new>` (+`--l1`) + `gtests.sh diff`; only the intended transitions |
| Behaviour/performance change | A/B differing is expected -> **measure**, but prove that untouched configs stay the same |
| Adding a test | On the code **before** the fix, the negative control FAILs as predicted |
- Tools/commands: chapter 05 5.2.3. If you changed the harness, rerun `selftest` (negative controls 5/5).
- **base is always an explicit SHA** (never HEAD: it moves when you commit). Never `--force` a snapshot. `corpus <dir>` deletes `<dir>/configs`, so do not regenerate the baseline corpora (r0, r1, r3, rT*) with a different base.
- ISA grades: A identical / B heavy ops identical + no scratch increase + inst ±0.5% (needs a performance measurement) / C anything else.
  Interpreting C: a ± in the `sync` count is classified as heavy -> if send/dpas/math/load/store/CALL did not move and no barrier was added, it can be cleared. A change in `disable_mid_thread_preemption` is IGC's ~600 inst threshold, not a functional change.
- `grep spill` is meaningless (`-abortOnSpill` is in every .asm). The real signals are zeinfo `*scratch*/*spill*` and `numGRF`.
- After a build, use `python3 test/sdpa_ocl_ab.py built --base <sha|snapshot>` to confirm your kernel made it into the `.inc` + `.so` + `ov_gpu_unit_tests` (only after linking finishes; the unit tests are statically linked, so they hold a separate copy).

## 2. Attribution rules -- violating them nearly led to "fixing" correct code
1. **An A/B differs in exactly one thing.** A one-sided control is a hypothesis, not a conclusion. "Standalone run + toggle" vs "sweep + no toggle" differ in two things (head-486 OOM incident: the cause was not the toggle but suite memory pressure).
2. **Run the toggle both on and off.** If the problem only appears in long runs, compare sweep vs sweep.
3. **A heterogeneous failure set = evidence**: if tests unrelated to the kernel are mixed in, diff the failure **sets** across configurations before building a theory.
4. **Check that the control is a real control**: e.g. on compressed BY_CHANNEL, `TEST_USE_SDPA_OCL(_DECODE)=0` was not a control back when there was a layout bug. Prefer provably no-op toggles (byte-identical, confirmed with ocloc).
5. **Producer before consumer**: the first bisection step for wrong output is to run the same input through a different backend (`TEST_USE_SDPA_OCL=0`). T5: sdpa_ocl was blamed, but the cause was the KV-cache quantization kernel (`dynamic_quantize` 256 cap).
6. **A metric moving != the metric mattering**: spill went 2688->0 yet it got 47% slower (the LSC message count was the real cost). occupancy% was inversely correlated with speed. Verify static models (ISA, SLM->occupancy) with on-device measurements (the SLM-occupancy hypothesis was falsified).
7. **Record it when the prediction is wrong**: in T7, "wrong unless the base is 64 B aligned" was a wrong prediction, found by running the negative control before the fix. Mark static-model predictions as PREDICTED.
8. Before any one-line verdict above, always check: is it the same binary, is it `--device_suffix=1`, did the env actually take effect (source dump or `SDPA_OCL_TRACE_CONFIG=1`).

## 3. Oracle/verification design
- **In a chained network, inputs are identical only at the first layer.** Layer averages measure propagated error (saturated values are identical for correct and buggy kernels). Compare the L0 error to the **output dtype ULP** (f16 ~4.9e-4 relative). Self-check: under causal, tokens `[0, tile)` must be bit-identical.
- **Toggle classification**: for bit-preserving toggles (same values/order: KV_2D/Q_2D/A_2D/DKS/256GRF), "no change" is a consequence of a correct kernel, not evidence. order-changing toggles (tile/per_wg knobs) moving a chaos-sensitive metric is a reroll of the dice -> "only the tile knobs move the score" is the signature of a correct kernel.
- **Never gate a kernel on end-to-end text metrics such as WWB** (±0.05). Use per-op gtests with an exact reference.
- **Check that the test data is sensitive**: N(0,0.1) makes softmax uniform and hides Q/K misreads (use `logit_scale_gain`, `runtime_scale_multiplier`). The LCG in `InputGenerateData` makes every row identical when `range*resolution` is a power of 2 <= the row length (confirm by offline replay). Compute whether the tolerance would accept an all-zero output.
- **NaN-poison fixture**: fill the output buffer with NaN and rerun to catch unwritten elements (the output buffer is not reset).
- **Real-model output oracle** (2026-10-23): llm_bench `Result MD5` per iteration is deterministic across runs; with `TEST_USE_SDPA_OCL_DECODE=0` GENERATE runs master's kernels, so equal to master in 20/20 (5 lengths x 4 iterations) means the prefill output is micro-identical. Always run a positive control (a known rounding change, e.g. the old tiling or `SDPA_OCL_MICRO_MATH=0`, must DIFF) and first check master vs master (qwen3.5/3.6: master varies between runs -> oracle void; fall back to new ⊆ master's set). Equality is not expected with a window/sink/token types (`MICRO_MATH` excluded).
- **Mask changes need uncompressed gtest rows**: a compressed cache relaxes the PA tolerance 2e-3 -> 25e-3 (u4 75e-3), and a dropped window term PASSed there (chapter 03 §2.7).
- **golden-split**: a run with the query range split == a single run (an equivalence oracle without a new golden).
- **sharp-softmax + negative-control switches** (`SDPA_OCL_NEG_*`, `SDPA_OCL_BIDIR=0`, `BIDIR_GATE=0`): turning them off **must FAIL**. If it PASSes, the suite does not observe the feature.
- If exhaustive enumeration is possible, enumerate: exhaustive check of the integer model (SWA 0..699 x seq 1..2099, etc.), brute-force pair checks (causal/window bound, 7.76M pairs).
- Never treat "SKIP/opt fallback = OCL success". The verdict is the dispatch census (>=1 line for the target kernel in `Enqueue stage ...` from `OV_VERBOSE=4`).

## 4. Execution environment trap check (skim once each)
- 2 GPUs: **`--device_suffix=1`** (B70). If omitted -> iGPU `sdpa_opt__*`, `TEST_USE_SDPA_OCL` ignored, `//.platform TGLLP`. (The DG2 PC numbers devices differently, `GPU.1` = HWQ 0.)
- Binaries: PA/SDPA unit = `ov_gpu_unit_tests`, `ScaledAttn*`/KV-cache subgraph = `ov_gpu_func_tests`. Wrong one -> `0 tests` (no error). `bin/intel64/{Debug,Release}/`.
- **`source setupvars.sh` wipes `$@` with `set --`**: capture `SDPA_ARGS=("$@")` before sourcing. If different configs show the same SLM/GWS/LWS, the toggle did not take effect.
- In the shell, do not pass `NAME=VALUE` through `"$@"`; use `env NAME=VALUE cmd`.
- For dumps: `OV_GPU_DUMP_SOURCES_PATH=./ OV_GPU_MAX_KERNELS_PER_BATCH=1` (single entry). `-device bmg` (xe2 is lnl-m). `NEO_CACHE_PERSISTENT=0` is sometimes needed.
- The `.inc` path is `build/src/plugins/intel_gpu/graph/impls/ocl_v2/codegen/include/` (no `src/`); long kernels are raw-string chunks.
- Tensor dumps: `OV_` prefix, trailing `/` on the path, the layer filter is a full regex_match on upper case (wrap it in `.*`).
- Debug binaries for accuracy, Release/RelWithDebInfo for performance. Even if the user says "I built it", confirm which tree.

## 5. Performance measurement/claim discipline
- For a large unexplained gap, check comparison validity before VTune/ISA output: actual dispatch/device, device memory, the final-linked reference binary, the same useful work and cadence. In the DG2 case in chapter 06, usm_host inputs and a wrapper-only micro binary each produced an invalid comparison.
- Metrics: label **kernel device time** (cliloader `-d -dv`) vs whole primitive vs e2e separately. SDPA prefill is ~10-15% of the 1st token, so 7% on the kernel = ~0.9% e2e.
- **Sweep prefill over prompt length (2K/4K/8K/16K/32K)**: attention's share of the first token grows with the square of the prompt (qwen3-8b 13% at 2K -> 66% at 32K), so a constant kernel gap becomes a growing e2e gap that a ~1K benchmark never shows (chapter 01 §6.6).
- **Check the clock pin at the start of every session** (`/sys/class/drm/card1/device/tile0/gt0/freq0/{min,max}_freq` = 2800): a reboot silently drops it, and the user asked for a full re-measure of everything taken unpinned. Record it in each summary.
- **Mixed-head models**: the tiling env overrides hit every head size and the encoder SDPA too (gemma-4: 256 + 512). Classify cliloader lines by heads (GWS1/LWS1) and per-hash call counts (`test/longctx/g4_an.py`); read the target group only.
- **Attribute e2e against the branch before the change** when master differs elsewhere (qwen3.6: master runs a `ref_matmul` the branch does not, 82-775 ms per prefill).
- B70 is **pinned at 2800 MHz**. For kernels with few calls, use **MIN/median over N runs** instead of the mean. Interleave the order of arms (ABBA); if the variance is larger than the difference, withhold judgment.
- Compare in **separate processes with the same source/build/inputs/cache state, differing in only one env** (env is cached within a process).
- Match the amount of work: ocl without the causal bound did 1.60x more work and was only 1.13x slower (compare macs/head and tile iteration counts before comparing efficiency). Fewer tile iterations != less time.
- **ocloc spill predictions differ from runtime spill** (4 candidates with 0 in ocloc spilled 3.3k-19k B at runtime). Choose candidates by runtime cliloader `SPILL=`. REG256 is not free (25% worse on the default tile; tune it paired with larger tiles).
- Static metrics (ISA inst count, SLM, occupancy%) are not performance evidence. Change only one thing per measurement, and keep falsified experiments in a table too (adopted/rejected + numbers).
- Hardware move: after B580 -> B70, absolute ns/ratios are STALE until re-measured. ISA/layout/accuracy remain valid. Label every number with device/clock/repetitions/metric/source, and distinguish MEASURED vs PREDICTED.
- The real-model harness template is chapter 05 5.5.1 + `test/run.sh`; in the performance-only mode `SDPA_SKIP_REF_CHECK=1`, accuracy needs a separate gate.
- Representative results/rejected experiments are in chapter 05 5.5.3-5.5.4 (e.g. u4 MIXED 145,603 vs micro 147,825 ns; 256GRF+tq32/pwk4/pwq2 = -7.0% vs micro, B70 pinned).

## 6. Wrap-up check
1. Tabulate the proof level/controls/negative controls, and do not hide prediction vs observation mismatches (`!!`).
2. State the regression risks: NaN/Inf, dynamic shape/shape inference, precision conversion, accumulator type, quantization, kernel selection, execution graph, memory changes (AGENTS.md).
3. Out-of-scope findings go only in Known issues. Code comments are a short "why"; measurements/derivations go in `docs/sdpa_ocl.md`.
4. Update memory + resume prompt + one-line commit proposal.
