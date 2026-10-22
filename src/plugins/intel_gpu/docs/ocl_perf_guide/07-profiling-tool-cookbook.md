# 07. Performance analysis tool cookbook (cliloader / VTune / GTPin / IGC / ocloc / harness / microbench)

This chapter collects in one place the tools and commands actually used in the DG2 case of chapter 06 (24x gap → within 3% of micro), their interpretation limits, and the traps we fell into. When investigating a performance problem of a new kernel, first look at "which tool for which question" and "what that tool does not tell you", then copy the commands.

All numbers are MEASURED on Arc A770 (DG2, xe_hpg, OpenVINO `GPU.1`, driver 26.27.39122.14) with VTune 2026.4. On other GPUs (Xe2/B70 etc.) use the same tools but re-check the numbers and the BDF. The raw data and scripts are in `test/sdpa_ocl_xe_hpg/s7a/perf/opt/` (untracked). This chapter abbreviates that path as `$O`.

```bash
repo=/home/shingyuk/work/openvino-eddy
O=$repo/test/sdpa_ocl_xe_hpg/s7a/perf/opt
```

## 7.0 Question → tool at a glance

| Question | 1st-line tool | What this tool does not tell you |
|---|---|---|
| Which kernels actually ran, how many times, with what geometry? | cliloader `-d -dv`, `OV_VERBOSE=4` dispatch census | Why it is slow |
| Pure GPU time of one call? | cliloader device time, the harness's OpenCL events | e2e latency, host overhead |
| Is there register spill/scratch? | cliloader `SPILL=`/`TPM=`, ocloc `.zeinfo` | Whether the spill *is the bottleneck* |
| Are the execution units idle, is DPAS used? | VTune GPU Hotspots `overview` (XVE active/stalled, XMX active, occupancy) | The source line/instruction of a stall |
| Where in the memory hierarchy is the blockage? | VTune `lsc-slm` / L3/SLM bandwidth, microbench ceilings | Latency of an individual load |
| Which basic blocks/instructions does the time concentrate in? | VTune `source-analysis` + GTPin (`bb-latency`, `mem-latency`, `instruction-count`) | Product latency, since instrumentation changes the schedule |
| What instructions did the compiler actually emit? | IGC ShaderDump, ocloc, iga64 | Execution frequency, latency hiding |
| Per-phase (KQ/softmax/SV) ratio inside one kernel? | `__builtin_IB_read_cycle_counter` timer | Time of the uninstrumented kernel (instrumentation changes the ISA) |
| DPAS/load/SLM ceilings of this hardware? | `mb/` microbench | Whether our kernel is limited by that ceiling |
| Is one change the cause? | Paired A/B in the harness + bidirectional controls | — (a discipline, not a tool; chapter 05) |

## 7.1 cliloader (CLIntercept): what actually ran

```bash
CL=/home/shingyuk/work/opencl-intercept-layer/install/bin/cliloader
source build/ov_install/setupvars.sh        # caution: setupvars wipes "$@", so save the arguments before source
TEST_USE_SDPA_OCL_HPG=1 PA_PERF_DEVICE_MEM=1 PA_PERF_ITERS=10 \
  $CL -d -dv bin/intel64/Debug/ov_gpu_unit_tests --device_suffix=1 \
  --gtest_filter='perf_paged_attention_prefill/*'
```

- `-d` records device time, and `-dv` also GWS/LWS (dispatch geometry). Early on `-d -ko` was used; later everything was unified on `-d -dv`. Recording the geometry lets you check immediately "whether it actually launched with the intended tile/subgroup settings".
- The min/avg/max in the output are over the calls within one process. Run different arms as **separate processes, in ABBA order** (environment variables are cached per process).
- **On a dGPU the input memory location changes the result.** The unit-test engine allocates inputs with `allocate_memory(layout)`, which defaults to `usm_host` on a dGPU, and the kernel reads them over PCIe. In A770 PA prefill, usm_host came out at 151.8 ms and usm_device at about 110 ms. Use `PA_PERF_DEVICE_MEM=1` (the test helper copies Q/K/V/cache to usm_device) and discard usm_host results. In this mode the reference cannot lock the inputs, so it is for perf tests only.
- Confirm which kernel was selected by its name: `sdpa_ocl_prefill_*`, `sdpa_micro__prefill_*`, `sdpa_opt_*`. If the tier gate rejects it, execution silently falls back to opt and the measurement is not an "OCL measurement". In the HPG experiments use `TEST_USE_SDPA_OCL_HPG=1`; the control is `TEST_USE_SDPA_OCL=0`.
- The `-d -dv` output was obtained with the Debug host library. Device kernel time is JIT-compiled OpenCL and hardly depends on the host build type, but a Release re-check remains a separate item to verify.
- **Read the real configuration back from `-dv`.** For `sdpa_ocl_decode` the line `SLM=... GWS[ 16 x A x P ] LWS[ 16 x SG x 1 ]` gives `SG_PER_WG` (LWS dim 1), the head groups (`A / SG`, from which M follows), and the partition count P (GWS dim 2); `SLM` cross-checks M. Aggregating per P bucket shows wave steps (e.g. 23.9 -> 35.4 us when 256 -> 272 workgroups). A knob can move a derived parameter (16 subgroups also raised M from 2 to 4), so do this for every arm before attributing.
- **Which cache did the model use?** 4-bit-weight PagedAttention models get a u4 KV cache by default. `OV_KV_CACHE_PRECISION=i8` forces i8; confirm that it took effect from the changed hash suffix of `pa_kv_cache_update_ref__<hash>` / `sdpa_ocl_decode__<hash>`.
- **Host-API census.** `cliloader -h` counts every OpenCL call (`clSetKernelArg`, `clFlush`, USM allocations, memcpy) per process; dividing by the decode steps compares two builds per token. An LD_PRELOAD shim that timestamps `clEnqueueNDRangeKernel`/`clFinish`/`clFlush` and splits at syncs gives the per-step kernel sequence (`test/ttft32/shim/libclshim.so`).

### Kernel source/binary dump (getting the reference kernel exactly)

```bash
OV_GPU_DUMP_SOURCES_PATH=/abs/dir/ OV_GPU_MAX_KERNELS_PER_BATCH=1 \
  TEST_USE_SDPA_OCL_HPG=1 PA_PERF_DEVICE_MEM=1 $BIN --device_suffix=1 --gtest_filter=...
```

- The path must be an **absolute path + trailing slash**. With `MAX_KERNELS_PER_BATCH=1` each kernel gets its own `.cl`/binary, which makes comparison easy.
- **sdpa_micro cannot be reproduced from the OpenCL source alone.** Recompiling the source wrapper drops the fused GEMM that nGEN splices in, and accuracy FAILs. The first binary that CLIntercept saves right after `clBuildProgram` is also from before the splice and FAILs. The comparison baseline is the **final-linked PREFILL native binary** (`*_<hash>_GPU.bin`, empty build options).
- PREFILL and GENERATE can share the same hash file (`B5C0FBCF`). Do not trust the hash alone; check the **kernel name inside the ELF** (`sdpa_micro__prefill_`). Using GENERATE as PREFILL gives an `argument-52` error or invalid timing. The harnesses (`run_extended.py`, `run_checked.py`) were fixed to refuse before the GPU launch if the PREFILL bytes are missing.
- When head/heads/kvheads change, micro's geometry (Q tile, SG count, SLM, spill) changes too. E.g.: h16/32 Q128/SG16, h48/64 Q128/SG32 (h64 spill 1,440 B, SLM 59,904), h80/96/128 Q32/SG16, >128 mostly Q32/SG32. Always get the baseline binary for a new shape by re-dumping the product.

## 7.2 Standalone harness: iterating on the kernel without a product build

Rebuilding OpenVINO takes several minutes each time and mixes in host code. Kernel design exploration was done in a **standalone OpenCL harness** (`$O/sdpa_bench`, `sdpa_bench.cpp`).

```bash
g++ -O2 -fopenmp -std=c++17 $O/sdpa_bench.cpp -o $O/sdpa_bench -lOpenCL
$O/sdpa_bench --verify --rows 96 --iters 8 \
  --arm c40=src:$O/v/c40_base.cl:TILED \
  --arm micro=bin:$O/micro.bin:micro
```

- arm syntax: `name=src:file.cl[:options...]` (runtime compile) / `name=bin:file:micro` (prebuilt native binary). `--shape seq,heads,kvheads,head`, `--warm`, `--iters`.
- To test the product kernel without a build, `assemble.py` combines the JIT header dump with the current product `.cl` into a runnable kernel. Re-assemble when the product code changes.
- Variant generation tools: `mkvar.py` (named source edits, including phase timer insertion), `tile.py` (rewrites the 8 tiling defines), `sweep.py` (valid-tiling sweep/ranking), `reparam.py` (shape rewrite).

Traps the harness created (each one actually happened):

| Trap | Symptom | Countermeasure |
|---|---|---|
| `#define TILED_NKT 256` in the variant source overrode the harness's `-DTILED_NKT=<NT>` | If seq does not match, the K'/V' addresses are wrong and `verify FAIL` (maxabs ≈ 4.9), but **the time looks normal** | `#ifndef` guard. Look at the verify line, not just the time |
| `--rows N` **samples the reference rows with replacement** | A "fullrow PASS" label was actually a sample | `--all-rows` mode (`sdpa_bench_checked`) and rc=2 on accuracy FAIL |
| The harness allocated and uploaded, outside timing, K'/V' buffers that are not used | Traces of the pre-pass remained in raw-only experiments | The raw-only harness (`sdpa_bench_extended`) refuses the allocation and the pre-pass run |
| `--prepass-src` runs the pre-pass only once and AB-measures only the kernel | Misread `prepass avg 0.000` as the "included time" | Measure the pipeline including the pre-pass separately |
| micro.bin is specific to rung-1 (h128, 32/8) | Used on other shapes it gives wrong/invalid results | Get a new final-linked binary from the product for each shape |
| Times of failed variants | A wrong kernel gets "benchmarked" as fast | Discard the times of accuracy-FAIL variants and keep only the log |

**Verify accuracy before timing.** Poison the output with NaN, compare against a CPU double reference for every row/head/channel (`maxabs < 1e-2`, no relaxing the tolerance), and separately check whether the result is **bit-identical** to the previous OCL variant (a change with the same math must be bit-identical). Use two input distributions: ordinary (Q/K/V std 1) and sharp (Q/K std √1.28, V std 0.1, scale 1/√head — matches the logit distribution of the product gtest's N(0,0.1)×gain128). Smooth data hides softmax/mask errors (chapter 03, chapter 05).

## 7.3 VTune GPU Hotspots

### Environment and basic commands

```bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
mkdir -p /tmp/vt                                  # without the parent directory the redirection fails before the GPU starts
vtune -collect gpu-hotspots \
  -knob gpu-profiling-mode=characterization -knob characterization-mode=overview \
  -knob target-gpu=0:3:0.0 -allow-multiple-runs \
  -result-dir /tmp/vt/NAME -- <command...>
vtune -report hotspots -r /tmp/vt/NAME -group-by computing-task -format csv
```

- `target-gpu` is a **PCI BDF**. On a 2-GPU system, check OpenVINO `GPU.1` (`--device_suffix=1`) and the VTune BDF separately (A770: OpenCL platform 0 / device 0 / `GPU.1` / BDF `0:3:0.0`). Profiling the wrong GPU still collects without any error.
- `$O/vt.sh <name> <mode> <command...>` wraps the above and produces a one-line CSV summary with `vtrow.py`. `mode` is `overview`, `lsc-slm`, `full-compute`, `instruction-count`, etc. `lsc-slm` needs `-allow-multiple-runs`.
- The progress log keeps overwriting one line with `\r`, so the output gets very large. Filter it with `tr '\r' '\n' | grep -v "Executing actions"`.

### How to read it (actual values from the chapter 06 DG2 case)

| Metric | OCL original | micro | Interpretation |
|---|---:|---:|---|
| XVE active / stalled | 15.6% / 81.8% | 44.2% / 55.6% | Execution units underused |
| XVE occupancy | 47.9% | 49.7% | Same level → occupancy cannot explain 24x |
| XMX active | 2.0% | 24.4% | DPAS barely used |
| L3 read bandwidth | 247 GB/s | 784–858 GB/s | Only a clue, not evidence of DRAM saturation |

- **Occupancy is not a verdict on speed.** At the same occupancy there was a 24x difference, and conversely, even at low occupancy (h128 seq257 MHA 5.6% vs 5.5%) the two kernels finished in the same time.
- **"XMX active" also counts latency.** A saturating DPAS microbench hits 100%, but our best kernel had an actual throughput of about 32% at XMX active 60% (cross-checked with phase timers/instruction counts).
- These metrics are *fractions of time*, not wall time. Time under profiling is longer than in a clean measurement (seq512, OCL/micro: 178/140 µs under profiling vs clean 137/123 µs).
- In the Xe2/B70 GPU Hotspots case there were 0 stall PCs, so nothing could be attributed to source lines/ISA instructions. In that case use the metrics only for *hypothesis generation* and adopt via A/B (chapter 04 §4.1.1).

### source-analysis + GTPin (basic block / memory latency / instruction count)

```bash
AMPLXE_MORE_GTPIN_OPTIONS='-allow_sregs 1' vtune -collect gpu-hotspots \
  -knob gpu-profiling-mode=source-analysis -knob source-analysis=bb-latency \
  -knob computing-tasks-of-interest='sdpa_ocl_prefill*#2#1#3' -knob target-gpu=0:3:0.0 \
  -result-dir /tmp/vtsa/NAME -- $O/sdpa_bench --iters 2 --warm 2 --arm NAME=src:variant.cl
```

(`$O/vtbb.sh`, `vtic.sh` (instruction-count), `vtrep.sh`, `vtsa.py` (joins GTPin events to native ISA PCs and aggregates per BB)).

- **Without `-allow_sregs 1`**, low-spill variants can return exit 0 yet have empty data ("kernels not found"). After collection, always check that real events and PCs were produced.
- `computing-tasks-of-interest` has the form `kernelNamePattern#callOrder...`. Instrumentation is slow, so run few iterations, e.g. `--iters 2 --warm 2`.
- **GTPin source-analysis failed on the sdpa_micro native with an assertion (`it != _insIdByOrigOffset.end()`).** The ELF function `st_size` was smaller than the actual fused `.text` length. It was reproduced by fixing only the symbol size in a profiling-only *copy*; the product native/fuser was not touched. Do not retry the same failure without new grounds. Overview characterization works normally on micro too.
- **GTPin's per-instruction cycles within a BB are a modelled distribution.** Trust only the BB sums. Ablating with synthetic operands (especially at 128 GRF) distorts the result through extra ALU/registers, so replacing the loads with SLM reads was cleaner.
- The instrumented build has a different schedule. Do not substitute instrumented results for acceptance timing.
- Representative conclusions obtained (A770, probe kernel): BB cycles were V loader 49%, K loader 28%; load ablation also gave K 7 ms, V 6.8 ms. These two independent pieces of evidence agreed, which established that "operand load is ~70% of the time".

## 7.4 IGC ShaderDump / ocloc / iga64: the ISA the compiler actually emitted

```bash
# dump the runtime compilation
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=/abs/dir  <run>
# offline (no GPU needed): with the same options as the runtime
ocloc compile -file build.cl -device dg2 \
  -options "-cl-mad-enable -cl-std=CL3.0 -cl-intel-256-GRF-per-thread ..." \
  -internal_options "-cl-intel-greater-than-4GB-buffer-required -cl-intel-has-buffer-offset-arg -cl-store-cache-default=2 -cl-load-cache-default=4"
```

- `$O/isa.sh <variant.cl> [G128]` → `/tmp/isa/<name>/` (`.asm`, `.zeinfo`); `isasum.py` summarizes the send/dpas/spill counts. **`isa.sh` changes the working directory and cleans up**, so check where it runs and what it targets.
- spill/scratch: the scratch in `.zeinfo` (the runtime value is cliloader `SPILL=`).
- `ocloc compile` **writes `*.bin/*.spv` into the cwd.** Running it from the repository root leaves stray files (`dpasw_peak_dg2.*` was once left in the repo root). Run it in a temporary directory or clean up afterwards.
- The `//.declare` header of `.asm` and `strings libigc.so` are huge. Narrow the output with grep.
- Offline ISA can differ from the runtime. Write conclusions only after confirming reproduction with the runtime options and the zebin actually loaded.
- Reading signatures (detailed table in chapter 04 §4.4): the `dpas` count and the instructions between them, `load.ugm.d32x8t` (block transposed), `send`/`sync.allwr` (waiting for load completion), `spill`/scratch fill, `goto/join` (dynamic branch).

### Examples of what was confirmed in the ISA

| Observation | Conclusion |
|---|---|
| IGC **sinks loads to right before their use**. At 128 GRF every step is `loads → sync.allwr → dpas`; even with full unroll at 256 GRF they are hoisted only about 20 lines ahead | Latency is exposed at every step. Moving a load earlier in the source may not change the ISA, so check with the ISA |
| In the original OCL, K was already a 32 B dword block read and only V repeated lane gather+pack | Corrected the "K/V both scalar gather" hypothesis. Look at the **executed ISA** first, not the source path |
| After limiting unroll, scratch fill ~104x lower, spill 7,872→704 B | The load count was unchanged, but spill reloads were a large part of the time |
| The full-tile guard cut dynamic goto/join by about 85.8% | 28→19 ms from control-flow cost alone at the same send count |
| `-cl-intel-no-prera-scheduling` is accepted but the ISA is identical | Do not record an ineffective option as "tried"; confirm with an ISA diff |
| `__asm__ volatile("":::"memory")` breaks block reads into scalar loads and causes spill | A compiler barrier is not free |
| Source variants with the same meaning varied by ±40%, and some 256-GRF fuse configurations were wrong | Verify accuracy for every variant. Keep the possibility of an IGC miscompile open |

## 7.5 In-kernel cycle counter phase timer

`__builtin_IB_read_cycle_counter()` works on DG2 (1 tick ≈ 1.08 ns). `mkvar.py timing/timing2` inserts phase timers into the kernel source and records per-WG ticks into a debug buffer (the 8th argument).

Phases of the c40 kernel (h128, seq4096) (ticks/iteration): K 5,068 / max 1,784 (barrier wait 607) / exp 2,533 (split wait 139) / SV 5,116, total 14.5K. In the ablation that replaced K'/V' with SLM reads, K −1,230 and SV −1,460 ticks → global latency ≈ 19%, SLM operand latency ≈ 15%, serial DPAS ≈ 2.7K per phase.

Limits (always observe them):
- **Instrumentation changes the ISA and the schedule.** For h64 seq1033, the instrumented kernel took 158 µs vs 152 µs uninstrumented; in the MHA gap, +1~8%. Use phase ratios only for *hypothesis generation*, not for acceptance timing or exact latency attribution.
- The phases mix in the waits of inactive SGs and barrier synchronization (the "max" phase looked like 53%, but that included inactive-SG waiting).
- Always confirm the effect of a fix derived from these values with a paired A/B of the *uninstrumented* kernel.

## 7.6 Microbench (`$O/mb/`)

`mb_run` (runner), `dpas_peak*.cl`, `ld_bw*.cl`, `slm_bw.cl`, `alu_ipc.cl`, `reorder.cl`, `cyc_run`. Used to gauge where our kernel stands relative to the hardware ceilings. A770 measurements:

| Item | Value |
|---|---|
| dpas.8x8 f16 | A thread has **only 1 in flight at a time** (~30 ns each, even with 8 independent accumulators). Saturates at ≥4 threads per EU: 128 GRF 10.05 ns/dpas = 104 TFLOPS, 256 GRF 9.04 ns = 116 TFLOPS. Operand values do not affect speed |
| dpasw | Same speed as dpas. The src1 sharing order also does not affect time → do not conclude that the opcode itself is a performance cause |
| L1 hit block read | 64 B messages 2.84 msg/clk/core, 256 B 0.93 msg/clk (~240 B/clk). L1 hit latency ≈ 137 ns |
| L3 stream | ≈ 4.1 TB/s (72 B/clk/core) |
| SLM block read | Up to 12.4 TB/s (215 B/clk/core) |
| GDDR | 2–3 TB/s |
| Occupancy | 256 GRF 4 threads/EU (occupancy ~50%), 128 GRF 8 threads/EU. No change at SLM 18–31 KB/WG, **halved at 47.7 KB** (SLM/WG is rounded up to 16/32/64 KB) |

Do not convert microbench throughput into overall attention speed. Microbenchmarks are for knowing the ceilings and message costs.

## 7.7 Timing discipline summary (common to all tools)

1. **Run only one thing at a time on a GPU.** If benchmarks, compiler dumps, VTune/GTPin, or native dump diagnostics overlap, the measurements in that interval are invalid. In practice an 8192 measurement accidentally overlapped a KQ-tail test and was excluded from acceptance. Confirm that the previous GPU process completed (`rc`) before running the next.
2. **Aggregate the same-round paired ratio.** Measure raw/micro alternately in the same round and use the median of the ratios. It can differ in the decimals from the ratio of separate median times; do not treat a value above 1.03 as PASS by rounding. Report **both** the per-arm medians and the paired ratio; do not pick just one.
3. **Look at both cadences.** `queued32/gap0` (32 kernels queued back to back) and `batch1/wait_each/gap200µs` (wait after every call). In wait_each the first call has a cold clock and cache and is strongly bimodal (20–40 µs vs 100–140 µs). Do not adopt a candidate that wins in only one of them. It was also confirmed that cliloader is not the cause (the same gap reproduces in the native harness without cliloader).
4. State the warm-up and round counts (e.g. queued warm6/48 rounds, gap warm64/256, final repeat warm8/64·warm100/512). Repeat with the seed and the arm order changed independently.
5. Accuracy before performance. Stop at an accuracy FAIL before timing, and discard the times of FAILed variants.
6. Keep the result files (JSON/log), the exact command, the source SHA256, and the micro binary used, together. Preserve failed/excluded runs with their reasons too (to prevent retries).
7. Bash `cd` persists in the session. Use absolute paths and run from the repository root. Filter large output (VTune progress logs, IGC `.asm` headers, `strings`).
8. Idle gap and clock: measuring the same kernel with gaps of 0/200/1000 µs showed a consistent +0.7~5.9% difference on some shapes, but a power/frequency cause was not proven. In that case record it only as "cadence sensitivity" and do not attribute it to power.
