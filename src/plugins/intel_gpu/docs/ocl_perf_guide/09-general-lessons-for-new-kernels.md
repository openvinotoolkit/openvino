# 09. General lessons for developing and optimising new GPU kernels

From the lessons of sdpa_ocl development (Xe2/B70 product path, Xe-HPG/DG2 standalone gate), this chapter collects only those **not specific to attention**. If the previous chapters are "what was done", this chapter is "what to do first in the next kernel (another operation, another architecture)". The supporting chapters are given in parentheses. Every quantitative value follows the hardware/MEASURED conditions of its chapter.

## 9.1 Bottleneck diagnosis decision tree

Symptom → what to check first → possible causes → how to confirm. A cause without a "how to confirm" is a hypothesis.

| Symptom | Check first | Possible causes | How to confirm (A/B) |
|---|---|---|---|
| **Several to tens of times** slower than micro/the reference | Does it really do the same work (kernel name, input memory location, reference binary)? | Comparison error (usm_host, wrong binary), opt fallback due to tier rejection | cliloader name and geometry, `usm_device`, final-linked binary (chapter 07 §7.1) |
| Large gap + XMX active ≈ 2%, large spill | Spill size (SPILL=), XMX active | Register pressure + huge unroll, extra accumulator | Limit the unroll → change in spill/scratch and time. Turn off options that add live accumulators, such as `MICRO_MATH` (chapter 06) |
| Gap + low XMX + small spill | Kind/count of load messages in the ISA | Lane gather (16 B), word-mov packing, block IO disabled due to alignment | Separate the gate conditions (legality), replace with block read (head-72: 9.28×, chapter 02) |
| `sync.allwr` between `dpas` every step | ISA order | IGC load sink → exposed latency | Change the load position/message grouping, then compare the ISA (native may be needed, chapter 08) |
| Wins at long lengths, **slow only at short lengths** | Tile padding ratio per length, WG count, fixed cost | Too few waves, prologue/epilogue, pre-pass fixed cost, waste of a large tile | Standalone A/B of a per-length `tiny` geometry (Q8/K64/128 GRF) (chapter 06 §6.3 "short lengths" section) |
| Wins in queued, **loses only in wait_each** (or the reverse) | Paired ratio per cadence, bimodality | Cold clock/cache, latency of the first K load | Measure the same kernel with gap 0/200/1000 µs. Do not conclude that power is the cause |
| Accuracy wrong **only at specific lengths/heads** | Partial tile, odd depth, tail band | Producer/consumer bound mismatch, reading unwritten SLM, clamp error | Boundary length scan + all-rows + NaN poison |
| Performance changes by ±tens of % regardless of source meaning | ISA diff | IGC schedule/allocation fluctuation, `inline` vs `always_inline` | Use a "no-op change" as a control via ISA SHA256 comparison (chapter 04 §4.8) |
| 256 GRF removed the spill but is slower | The kernel's dominant cost (e.g. gather) | Halved occupancy + higher load cost | Measure GRF/tile/SG count together (chapter 04 §4.5) |

## 9.2 Where the gains came from (a sense of the ordering of effect sizes)

DG2 PA prefill (h128, 32/8, seq4096, f16, causal): 126.7 ms → 4.25 ms (micro 4.52 ms). Each step is a separate matched A/B, and the values come from different runs.

| Order | Change | Effect | Category |
|---:|---|---|---|
| 1 | Limit the S*V loop unroll | 126.8 → 57.6 ms | Register pressure/spill removal |
| 2 | Limit the KQ loop unroll | 57.6 → 51.0 ms | Same (spill 7,872 → 704 B) |
| 3 | Move the V boundary outside the full-tile guard and read 64 B rows | 51 → 28.2 ms | Gather → block read |
| 4 | K full-tile guard | 28.2 → 19.0 ms | Removed 85.8% dynamic goto/join |
| 5 | `MICRO_MATH=0` (remove the second accumulator) | 18.9 → 11.8 ms | Live range |
| 6 | Tile-major K'/V' (pre-pass) | 11.8 → 5.1 ms | DPAS operands via 256 B block reads |
| 7 | 256 GRF + c40 tile (KQ 16×32, SV 32×16) | 5.1 → 4.25 ms | Tile/GRF/occupancy |
| 8 | Remove the pre-pass: read raw K/V inside attention and pack natively (asm) | Pre-pass fixed cost removed | Structure (chapter 08) |
| 9 | Per-head and per-length geometry/policy (tiny, key8, Q16, clamped SV, …) | Closed the gap at short lengths/small shapes | chapter 06 §6.3 "short lengths" section |

**Generalisation:** the first two-digit factors came from (a) removing comparison errors, (b) register pressure/spill, (c) the memory message form (gather vs block). "Textbook optimisations" such as tile tuning or prefetch were within the last 10%. For a large gap, look in the order **measurement reliability → spill → message form** rather than textbook tuning.

## 9.3 Design principles (repeatedly confirmed in this project)

1. **Message form first, tile later.** Whether the same bytes are read as 4 × 16 B or as one 64 B read mattered more than the tile size (head-72 9.28×, h128 V read 51→28 ms). The legality conditions (alignment/pitch/width) must be separated precisely to get block IO back (chapter 02).
2. **Design the live ranges.** Even if a large tile is faster, two sets of accumulators lose to spill (A_tile1 64 GRF of `MICRO_MATH=1`). Draw which values are live across which loop. 256 GRF pays with halved occupancy (4 vs 8 threads/EU), so first check "does the needed tile require 256?".
3. **One value for the producer/consumer bound.** For a tile passed through SLM, if the width the producer wrote differs from the width the consumer reads, unwritten SLM is read (key8: `min(WGK, causal_k-k0)`). Test tails and partial tiles separately.
4. **Where SLM ownership changes, a WG barrier is needed.** When K and P scratch were aliased, a missing barrier for all readers to finish at the ownership change was a race (maxabs ≈ 0.9). The assumption "the same subgroup writes and reads, so a subgroup barrier is enough" was also checked by A/B first.
5. **With branches/skips, control flow changes code generation.** Reducing the number of tail iterations differs from removing the tail code path. "Always run everything and mask" can be faster, as with clamped whole-SV (branch and marshalling removed). Conversely, the KQ early-exit produced NaN in some lowerings. Either way, near synchronisation it was safe to put the uniform branch inside asm.
6. **Small shapes are a separate kernel configuration.** Too few waves, the prologue and the first-load latency dominate, so a "tiny" geometry such as Q8/K64/128 GRF, shrunk from the large tile (Q32/K128/256 GRF), was needed. Using the long-length configuration as is at short lengths is slower than micro (normal clamp lost at h128 seq1/16/32/64, and at 127 tiny lost by +6.05% — the policy must switch at the boundary) (chapter 06 §6.3 "short lengths" section).
7. **The best geometry differs per head/dtype/layout.** Instead of choosing one fixed tile, build the policy from explicit selection arguments such as `(head, Qheads, KVheads, max_len)`. Multiple sequences are selected by **the maximum length, not the summed length**.
8. **Generalise by align/depth/active-group conditions, not by head size.** head80 (5 depths, odd), head48 (unaligned to the V tile width) and head272 (partial band) each broke in a different way. Rules such as "skip padded value subgroups" and "skip the absent upper band with a uniform branch" were the general solutions.
9. **Solving it with a pre-pass is a temporary measure.** An external layout rearrangement (K'/V') sped up the long-length diagnosis, but its fixed cost (about 44–110 µs) loses at short lengths. If the final goal is "replacement", it must be solved inside attention without a pre-pass and without a hybrid fallback (the user's decision).
10. **Confirm the hardware facts before writing assumptions.** Values such as 1 DPAS in flight per thread, L1 prefetch being a loss, dpasw ≈ dpas, and SLM 47.7 KB halving occupancy narrowed the design choices a lot. On a new architecture, run the microbenchmarks of chapter 07 §7.6 first.
11. **A static tiling choice is a compromise over runtime shapes.** The best decode `SG_PER_WG` flips with the partition count (i8: 16 subgroups 0.85-0.89x at short context, 1.26-1.29x at 4K). Model it (threads per partition against the device's thread count, per-wave time), pick the side with the smaller worst-case loss, and record the per-dispatch variant as the way to get both.

## 9.4 Accuracy verification discipline

- **Accuracy before timing.** A wrong kernel also gets "benchmarked" fast (K'/V' with wrong addresses). Discard the times of FAIL variants.
- **All-rows, NaN poison, fixed tolerance.** Sample-row verification (sampling with replacement) is not "fullrow". Filling the output buffer with NaN exposes unwritten regions immediately. Do not relax the tolerance for performance (`maxabs < 1e-2`).
- **Bit-identical comparison.** A change that does not alter the math must produce the same full output as the previous kernel. If it does not, the order changed, so check whether that was intended (key8 differs in a few halfwords because of different reduction grouping — recorded, then accepted).
- **Sharp input stress.** Smooth data (N(0,0.1)) hides Q/K misreads and softmax errors. Always use sharp inputs (or a gain) that match the logit distribution (chapter 03, chapter 05).
- **Check for degenerate data.** If the LCG range/resolution is a power of 2, every row can end up with the same values. Confirm offline first that the data has meaningful variance.
- **Boundary scan.** Lengths: 1, 16, 31–33, 63–65, 96/97, 127–129, 255–257, 511–513, 1024/1025/1033, 2048, 4096, 8192. Head: odd depth, unaligned band. Also multiple subsequences and padding (Q before/after, NaN physical pad).
- **A PASS at one length does not revive a candidate.** There really was a candidate that passed only seq256 and gave NaN at 257.

## 9.5 Experiment and result management discipline

1. **Freeze the fixed policy first, measure afterwards.** Picking only the candidates that looked good after measuring them (cherry-pick) makes the gate meaningless. Fix the selection policy and the source SHA256 before measuring, and stop at the first FAIL (`run_policy.py`).
2. **Change only one thing at a time.** The time of a candidate combining many structural changes is only a total effect, not evidence of a single cause. Attribute only with controls on both sides ("no attribution from a one-sided control", chapter 05). Avoid the double change of changing the source but missing a compile flag, and the mistake of comparing builds that only share a name.
3. **No-op/branch-only control.** Keep a control that only adds the branch without reducing work, such as `DIAG_TAIL_LIMIT=k`, and a control whose source changed but whose feature is off (confirm the final ISA SHA256 is identical).
4. **Repeats must be independent.** Change the seed, the arm order (reverse) and the cadence. Do not adopt a small improvement (≤2%) before confirming it with repeats.
5. **Do not pass by rounding.** Against a 1.03 criterion, 1.0306 is a FAIL. One tiny win does not replace the whole gate.
6. **Record failures.** For a discarded candidate, leave *why* it was discarded and *what must not be retried* in the HANDOFF dead-end list (spill block variant, sweeps, KQ skip, wide SLM load, etc.). Retry when new evidence appears.
7. **Do not hide overlap accidents.** If GPU jobs may have overlapped, exclude those measurements, close with a clean repeat, and record the accident too.
8. **Distinguish the layers of evidence when you speak.** (a) standalone harness PASS (b) product dispatch confirmed (c) product accuracy gtest (d) e2e latency (e) regression gate (S6/SG16/B70) are different proofs. Do not write a harness PASS as "product replacement complete". Write coverage holes (number of heads checked, head-count, padding, feature) in the same document as the PASS table.
9. **Metrics are clues; the verdict is the A/B.** Spill bytes, occupancy, XMX active, bandwidth, instruction count and IPC are each independent clues, and a conclusion that a change in one of them caused a speed change must be supported by a matched device-time A/B. Example: the kernel with fewer instructions (920) was not always faster than the larger one (1585, micro), and IPC was low at 0.22–0.24, so the time came from *latency/dependencies*.

## 9.6 Collected tool/environment pitfalls

| Pitfall | Result | Rule |
|---|---|---|
| dGPU input in `usm_host` | Read over PCIe, so the kernel is slower (about 38% in PA prefill: 151.8 vs 110 ms) | Copy to `usm_device` (chapter 07 §7.1) |
| Recompiled micro source wrapper / initial clBuildProgram binary | nGEN GEMM missing → accuracy FAIL, time invalid | Final-linked PREFILL binary, check the ELF name |
| PREFILL measured with the GENERATE binary | Argument error `argument-52` | The harness rejects it if there are no PREFILL bytes |
| `--device_suffix` missing on 2 GPUs | iGPU/sdpa_opt runs, so it is not an "OCL measurement" | dGPU `--device_suffix=1`, check the dispatch name |
| `setupvars.sh` wipes `"$@"` | Script arguments disappear | Save the arguments in a variable before sourcing |
| A Bash `cd` persists in the session | Relative-path scripts run in the wrong place | Absolute paths, fixed repository root |
| `ocloc compile` writes `*.bin`/`*.spv` into the cwd | Stray files in the repository | Run it in a temporary directory |
| GTPin without `-allow_sregs` exits 0 + no data | Mistaken for a successful collection | After collecting, confirm that events/PCs exist |
| Parent of the VTune result dir missing | The redirection fails before the GPU starts | `mkdir -p` |
| Editing a running script | bash resumes at the old byte offset | Check whether it is running before editing |
| Debug host + DEBUG_CAPS | e2e/host overhead values are unreliable | Use only device kernel time, or re-measure with Release |
| Instrumentation (phase timer/GTPin) used for acceptance | The ISA changes, so the time differs | Instrumentation is for generating hypotheses |
| default configuration not checked (KV-cache precision) | a rule measured on u4 was shipped for i8 and regressed it | state the cache actually used (4-bit weights -> u4); measure each cache type |
| a knob moved a derived parameter (SG -> M through the register estimate) | a confounded A/B credited SG with M's effect | read LWS/GWS/SLM back for every arm |
| arm names that are prefixes of each other in result globs | one arm's statistics included another arm | glob `<arm>_r[0-9]*`, check the sample counts |

## 9.7 Checklist for starting the next kernel

1. Define one call: operation, dtype, shape, layout, mask, device/driver, build, timing API, cadence. Fix the reference kernel (binary).
2. Confirm that the comparison does the same work (kernel name, memory location, useful FLOPs and the real key count/padding).
3. Collect runtime evidence first: device time + geometry, SPILL/TPM, SIMD/GRF/SLM. Build a length/shape scan table.
4. If the gap is several-fold, follow §9.1 from the top. Use a profiler tied to one question (chapter 07 §7.0).
5. Write a falsifiable prediction: "replacing this reader's gather with a block read lowers the time for the same bytes".
6. One change, controls in both directions, accuracy → performance. Confirm that the JIT constants, the final binary and the ISA signature before/after the change differ as intended.
7. Widen the performance gate: short/long, boundaries, head-count, two cadences.
8. Leave the results, failures, coverage holes and reproduction commands (HANDOFF/CHECKPOINT). Notes together with hashes.
9. Product integration is a separate step: dispatch confirmation (census), accuracy gtest (including sharp), SG16/other-architecture byte-identity, e2e.
10. If asm/native is needed, put chapter 08's checklist and a post-integration maintainability review into the plan.
