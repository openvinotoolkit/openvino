---
name: ocl-kernel-performance-investigation
description: Investigate large or unexplained Intel GPU OpenCL kernel performance gaps using runtime timing, VTune/GTPin, compiler ISA, and controlled attribution. Use before tuning when the bottleneck is not yet established.
---

# OpenCL GPU kernel performance investigation

Use this skill when a GPU OpenCL kernel is unexpectedly slower than a reference kernel, has a large unexplained regression, or when profile counters disagree with elapsed device time. It focuses on diagnosis; use ocl-kernel-ab-methodology to prove a proposed change and xe-spill-isa-profiling for deeper scratch/ISA analysis.

Tool commands and caveats: [07-profiling-tool-cookbook.md](../../../src/plugins/intel_gpu/docs/ocl_perf_guide/07-profiling-tool-cookbook.md) (skill `xe-profiling-tools-cookbook`); inline vISA: [08](../../../src/plugins/intel_gpu/docs/ocl_perf_guide/08-inline-visa-asm.md) (skill `xe-visa-inline-asm`); generic lessons: [09](../../../src/plugins/intel_gpu/docs/ocl_perf_guide/09-general-lessons-for-new-kernels.md) (skill `ocl-kernel-dev-lessons`). Case evidence and the SDPA investigation are in [06-performance-gap-investigation.md](../../../src/plugins/intel_gpu/docs/ocl_perf_guide/06-performance-gap-investigation.md). Read [05-methodology-and-pitfalls.md](../../../src/plugins/intel_gpu/docs/ocl_perf_guide/05-methodology-and-pitfalls.md) for benchmark discipline and [04-spill-isa-profiling.md](../../../src/plugins/intel_gpu/docs/ocl_perf_guide/04-spill-isa-profiling.md) for spill, VTune, and ISA mechanics.

## Establish a trustworthy comparison

1. Define one exact workload: device and driver, operation/stage, shape, dtype, layout, mask, memory placement, input seed, build, warm-up, repetitions, and enqueue cadence.
2. Prove which kernel ran on each arm. Check dispatch census or runtime kernel names; do not count opt fallback, skip, or a failed JIT as a successful OCL run.
3. Check the actual device and buffers. On multi-GPU systems verify the OpenVINO device suffix and profiler GPU BDF. For a discrete GPU, establish whether inputs are in device memory; host-resident buffers can make PCIe transfer behavior look like a kernel problem.
4. Verify the reference binary. Some generated kernels have a final link/fusion step. A source wrapper or intermediate binary can omit native code and produce invalid timing or output.
5. Compare device kernel time, primitive time, and end-to-end latency as separate metrics. If the workload executes little work, report distributions or paired ratios rather than a single mean.
6. Serialize GPU measurements. Do not overlap benchmark runs, compiler dumps, or VTune/GTPin collection on the same device; use fresh result directories and discard runs with uncertain dispatch or interference.

## Choose the measurement for the question

| Question | Useful evidence | Limit |
|---|---|---|
| Which kernel ran, how often, and how long did its device events take? | CLIntercept/cliloader device timing and kernel-name geometry | Does not identify a source-level bottleneck; profiling can add host overhead |
| Is the GPU underused or spending time stalled? | VTune GPU Hotspots: XVE active/stalled, occupancy, XMX, memory metrics | Occupancy is not throughput; stalled percentage is not a fraction of kernel wall time |
| Are scratch, gathers, branches, or DPAS instructions in the hot path? | Runtime spill fields, IGC ShaderDump, ocloc with matching options, iga64/native disassembly | Static instruction counts do not give dynamic frequency or latency hiding; offline output must match runtime |
| Which native region sees loads or latency events? | VTune source-analysis/GTPin with PC mapping to the exact loaded zebin | Discard runs with missing kernels, invalid trace, mismatched text, or stale symbol metadata |
| Is a hardware component a plausible limit? | Focused DPAS, global-memory, SLM, or ALU microbench | Component peak is not whole-kernel performance |
| Which phase is expensive in one build? | Diagnostic cycle-counter phase markers | Instrumentation can change ISA and scheduling; never use it as clean acceptance timing |

For VTune, record version, driver, Metrics Discovery version, device BDF, result directory, selected kernel, profiling mode, and event names. A nonzero exit code is not the only validity check: confirm that the result contains the intended GPU metrics and PC records. In the recorded A770 session, GTPin needed SREG allowance for low-spill variants; an exit-zero collection could otherwise contain no kernels. A fused micro binary also had an ELF function size smaller than its final native text, so source-analysis required a profiling-only metadata correction. The native text and product fuser were left unchanged.

## Typical order that worked (DG2 24x gap, step sizes MEASURED)

1. Fix the comparison (usm_device inputs, final-linked reference binary, kernel name): removed a false 38% (usm_host).
2. Runtime spill + VTune overview: XMX 2% vs 24%, XVE stalled 82%, spill 7,872 B -> candidates, not causes.
3. ISA of the executed kernel: scratch fill dominated; only V (not K) was a lane gather -> corrected the hypothesis.
4. Source-injection ABBA, one factor each: unroll limits 126.8 -> 51.0 ms, V block read -> 28.2, K full-tile guard -> 19.0 (goto/join -85.8%).
5. Standalone harness: kill the second accumulator (18.9 -> 11.8), then GTPin bb-latency + load ablation put ~70% in operand loads; 256 B block reads of tile-major K'/V' -> 5.1; tile/GRF scan -> 4.25 (micro 4.52).
6. Length scan exposed short-length and pre-pass costs; each hypothesis (causal tail, wave count, idle gap, DPASW, micro packing) was refuted or confirmed alone; final design read raw K/V inside one kernel with native packing.
7. Close with a frozen-policy gate over lengths x head-counts x both cadences.

## When the ratio changes with problem size (B70 head 512 case, chapter 06 §6.7)

1. Clean the axis: inputs from one source (prompts cut from one text), time normalised by the expected growth (keys^2 for causal attention); find the knee.
2. Check the clock that was reached, not the pin: sample `act_freq` + `throttle/reasons`; PL2 held a 2800 MHz pin at ~2150 MHz in 32K prefills. Repeat the sweep at a fixed clock below the limit (1800 MHz). There the reference was flat and only our kernel kept the knee (+9.6%), and the ratio drift was unchanged: the clock slowed both, the knee alone moved the ratio.
3. Match the knee to a working set crossing a cache size (one KV head's K+V, 2 KiB per token, vs the 24 MiB L3 at 12K tokens); profile both ends at the fixed clock (L3 miss 3% -> 18%, Send stall 20% -> 30%, DRAM read far under peak = latency).
4. Falsify with value-preserving changes, so a bit-identical output check covers correctness: a workgroup reorder (no effect -> not reuse), an existing prefetch (knee gone -> miss latency), then the prefetch's cost model (stride 64/128/256 B -> cost = line requests).
5. Adopt the narrow form and measure where it does not apply (here: 1-5% slower at head <= 256, kept off).

## When the gap to the reference grows with the prompt in decode (chapter 06 §6.10)

1. Reproduce both arms under one pinned clock, interleaved, with the load logged; then split the decode step per kernel: two cliloader runs with `-ic 1` and `-ic 129`, per-family totals subtracted and divided by runs x 128 (`test/g4dec/cli_pair.sh`, `dec_an.py`). gemma-4-26b at 32K: everything matched except the head-512 decode call (712.8 vs 441.9 us).
2. Read the configuration off the dispatch line before theorising: `GWS dim 1 / LWS dim 1` = head groups -> M = 1 at kv group 8, so each page was read and dequantized 8 times per kv head (the reference: 2).
3. Sweep the existing env knobs (M, 256 GRF) before writing code: larger M won even while spilling -> registers were the limit.
4. Iterate in the standalone probe once it reproduces the in-model per-call time (679 vs 714 us, 630 vs 624), and keep every step bitwise against the baseline.
5. Bench every shape the new rule reaches (all capped decode shapes of the daily models, u4 and i8) before shipping it; that found and excluded the head-64 / i8 short-context loss.

## When utilisation matches but time does not (chapter 06 §6.8)

- Compare per-type instruction totals with the reference (VTune overview csv: `XVE Instructions:XMX instructions`, ALU0/ALU1). Short-prompt PA PREFILL: XVE active 63% / 64%, XMX busy 58% / 59%, but XMX instructions 2.49 G vs 1.85 G = redundant work (the masked causal triangle), fixed by a per-subgroup skip (qwen3-8b 1K 253 -> 216 us, micro 214).
- Bound a candidate with the ideal variant first (token-major page for the f16 d-major read), then remove confounds shared by both arms before attributing the rest (the Kc branch moved both arms ~10%).

## When a compiler upgrade made a kernel slower (chapter 06 §6.9)

- Re-measure the baseline under the installed driver, then read `SPILL=`/`TPM=` per kernel. The same `TPM=` on every shape with no `//.spill size` is a private array: name it from the `alloca`s left in `*_optimized.ll` (the KV-cache writer: two helpers' `[2][9]` arrays in one slot, uniform runtime indices no longer promoted by IGC 2.41).
- Calibrate a standalone probe against cliloader at the real dispatch size before trusting it (512-block PREFILL matched; small GENERATE ratios did not).
- After the fix, sweep every shape again: removing the scratch exposed a store-order contention at head 256 (+7%) that the reloads had hidden.

## Turn evidence into a testable cause

- Inspect the executed native instructions before labeling a path. In the DG2 SDPA case, an initial “K and V are both gathers” hypothesis was wrong: K already used a 32 B dword block read while V performed lane gathers and packing.
- Separate spill from other private memory. Runtime SPILL/TPM, compiler scratch metadata, and the actual scratch load/store sequence answer different questions.
- Do not optimize a single summary metric. 256 GRF removed spill but lost performance in some B70 cases; high occupancy coexisted with stalls; memory bandwidth did not by itself prove DRAM saturation.
- Write one falsifiable hypothesis and its predicted observation. Change one factor, keep a matched control, and verify the config/binary actually changed. Source-injection or offline experiments are diagnostic until the product path and accuracy gates pass.
- Treat data movement geometry and work volume as first-class. Count executed messages, useful keys, padded/tail work, SLM traffic, and producer/consumer bounds instead of inferring cost from source statements.
- If vISA inline assembly is needed, record its source hash, virtual-register shape, operand mapping, target/compiler version, and generated native text. The compiler still assigns physical GRFs; manually written assembly does not remove liveness, clobber, mask, tail, or compiler-version risks. Validate a product integration separately from a standalone asm harness.
- Re-run exact output checks before timing. Include all rows/channels, NaN/Inf, partial tiles, boundaries, and a fixture that exposes Q/K mistakes.

## Close the investigation honestly

Report the measured baseline, the strongest supported cause, the A/B that supports it, rejected hypotheses, and remaining coverage. Mark every result as measured or predicted, and include the exact device, input, binary/source hash, options, repetitions, cadence, and metric.

Keep the proof levels separate:

- A profiler result identifies a candidate cause.
- A controlled device-time A/B attributes a change on that workload.
- A shape/cadence matrix establishes the tested kernel gate.
- Product dispatch, product correctness, and end-to-end measurements establish integration.

The DG2 case reached the user-approved standalone gate of OCL <= micro × 1.03 across its recorded matrix, with a worst paired ratio of +2.6234%. That threshold belongs to that experiment. The measured candidate uses vISA inline assembly to read and pack raw K/V inside one attention kernel. An older `KV_TILED`/pre-pass product prototype is a separate, superseded path and does not inherit the raw-kernel result. Product raw-kernel integration, end-to-end result, S6/SG16/B70 regressions, and broader input coverage were still open; do not describe the standalone gate as a completed product replacement.
