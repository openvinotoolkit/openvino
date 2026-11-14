---
name: xe-profiling-tools-cookbook
description: Commands and interpretation limits for Intel Xe/Xe-HPG GPU performance tools in OpenVINO intel_gpu - cliloader (-d -dv), usm_device inputs, final-linked micro binary dump, VTune GPU Hotspots (overview/lsc-slm/source-analysis), GTPin bb-latency, IGC ShaderDump/ocloc/iga64, cycle-counter phase timers, microbench ceilings, standalone OpenCL harness. Use when you must actually collect or read profile data for a GPU kernel.
---

# Xe GPU performance analysis tool cookbook

Full explanations and numbers: `src/plugins/intel_gpu/docs/ocl_perf_guide/07-profiling-tool-cookbook.md` (this skill is a summary). The *investigation procedure* for large gaps is in `ocl-kernel-performance-investigation`; proof/attribution discipline is in `ocl-kernel-ab-methodology`.

## Question → tool

| Question | Tool | What it does not tell you |
|---|---|---|
| Which kernel ran with which geometry / device time | `cliloader -d -dv`, `OV_VERBOSE=4` census | the cause |
| spill/scratch | cliloader `SPILL=`/`TPM=`, ocloc `.zeinfo` | whether it is the bottleneck |
| Execution unit·XMX utilization | VTune GPU Hotspots `characterization-mode=overview` | source lines |
| Memory/SLM bandwidth | VTune `lsc-slm` (`-allow-multiple-runs`) | individual load latency |
| BB/memory latency, instruction count | VTune `source-analysis` bb-latency/mem-latency, `instruction-count` + GTPin | product latency under the instrumented schedule |
| The ISA the compiler produced | IGC ShaderDump, ocloc, iga64 | execution frequency·latency hiding |
| Per-phase ratios | `__builtin_IB_read_cycle_counter` (DG2 1 tick ≈ 1.08 ns) | time of uninstrumented kernels |
| Hardware ceilings | microbenchmarks (dpas/ld/slm/alu) | whether the whole kernel is capped by a ceiling |
| Did the GPU run at the pinned clock? | `act_freq` + `throttle/reasons` sampler (`test/longctx/freq_sample.py`, chapter 07 §7.8) | which kernel ran at which clock (whole-run means mix kernels) |

## Key commands

```bash
CL=/home/shingyuk/work/opencl-intercept-layer/install/bin/cliloader   # always -d -dv
TEST_USE_SDPA_OCL_HPG=1 PA_PERF_DEVICE_MEM=1 PA_PERF_ITERS=10 $CL -d -dv $BIN --device_suffix=1 --gtest_filter='perf_paged_attention_prefill/*'
# source/binary dump (absolute path + trailing slash)
OV_GPU_DUMP_SOURCES_PATH=/abs/dir/ OV_GPU_MAX_KERNELS_PER_BATCH=1 ...
# VTune (pin the GPU by BDF, mkdir the result directory's parent in advance)
source /opt/intel/oneapi/setvars.sh; mkdir -p /tmp/vt
vtune -collect gpu-hotspots -knob gpu-profiling-mode=characterization -knob characterization-mode=overview -knob target-gpu=0:3:0.0 -allow-multiple-runs -result-dir /tmp/vt/NAME -- <cmd>
vtune -report hotspots -r /tmp/vt/NAME -group-by computing-task -format csv
# GTPin source-analysis requires -allow_sregs; minimize the call count
AMPLXE_MORE_GTPIN_OPTIONS='-allow_sregs 1' vtune -collect gpu-hotspots -knob gpu-profiling-mode=source-analysis -knob source-analysis=bb-latency -knob computing-tasks-of-interest='kernel*#2#1#3' ...
# IGC dump / offline ISA (ocloc writes *.bin/*.spv into cwd → run it in a temp directory)
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=/abs/dir <run>
```

## Must-follow rules

1. **Inputs must be `usm_device`**. On a dGPU the unit-test engine defaults to `usm_host` (PCIe) → 151.8 vs 110 ms. `PA_PERF_DEVICE_MEM=1`.
2. **The sdpa_micro reference is the final-linked PREFILL native binary**. A wrapper recompile/initial clBuildProgram binary FAILs because the nGEN GEMM is missing. The same hash file also contains GENERATE, so check the **kernel name inside the ELF** (`sdpa_micro__prefill_`). If head/heads change, the micro geometry changes too, so dump the product again for a new shape.
3. 2-GPU: check OpenVINO `--device_suffix` and the VTune `target-gpu` BDF **separately**. A770 = OpenCL p0/d0 = `GPU.1` = BDF `0:3:0.0`.
4. VTune interpretation: occupancy is not a speed verdict (24x difference at the same occupancy; conversely, a tie even at ~5% occupancy). "XMX active" also counts latency (looks like 60% when actual throughput is ~32%). All ratios are time ratios, not wall time. If there are 0 stall PCs, do not attribute to source.
5. GTPin: without `-allow_sregs 1` it exits 0 but reports "kernels not found" with no data → after collection, confirm events/PCs exist. For the sdpa_micro native binary, the ELF `st_size` is smaller than the fused `.text`, so an assertion fails → modify only a *copy* for profiling, leave the product native binary as-is. Trust only BB totals (the distribution within a BB is a model).
6. Cycle-counter instrumentation changes the ISA (+1~8%). For hypothesis generation only; no acceptance timing.
7. **Only one GPU job at a time**. Overlapping benchmarks/dumps/VTune invalidate that measurement.
8. Judge by the median of same-round raw/ref paired ratios. Both queued32/gap0 **and** batch1/wait_each/gap200. Report both the arm median and the paired value. Never pass a threshold by rounding.
9. Output hygiene: VTune progress log `tr '\r' '\n' | grep -v "Executing actions"`; do not print the `//.declare` headers of IGC `.asm` or `strings libigc.so`. No Bash `cd` (absolute paths). `setupvars.sh` wipes `"$@"`, so save the arguments before sourcing.
10. Scripts like `isa.sh` change cwd/clean up, so read them before running. Do not edit a running script.
11. **The clock pin is a request** (B70, 2026-10-24): under a sustained load PL2 lowered a 2800 MHz pin to 2133-2167 MHz in 32K prefills (`throttle/reasons` = `pl2`, 68-78% of busy samples; 8K 4%). Sample `tile0/gt0/freq0/{act_freq,throttle/status,throttle/reasons}` during long runs; for comparisons across sizes or across layers that run at different times, have the user pin below PL2 (1800 MHz gave 0 `pl2`) and verify. A one-length A/B at the pin stays fair.
12. **VTune on B70 / xe**: permission error -> `sudo sysctl dev.xe.observation_paranoid=0` (the message only names i915); BDF `0:3:0.0`. VTune strips the double quotes of target arguments, so pass JSON (llm_bench `--cb_config`) as a file; `-knob computing-tasks-of-interest=sdpa_*`, llm_bench `-n 1 -ic 1`; `test/longctx/vt_run.sh` / `vt_an.py`.
13. **Compare instruction totals, not only utilisation**: the overview csv carries `XVE Instructions:{ALU0,ALU1,XMX,Barrier and Sync}` per computing task; equal XVE active / XMX busy with +35% XMX instructions meant redundant work (chapter 06 §6.8). Grep a dump's `.cl`/`.asm` with `command grep` (the agent shell's `grep` is a ugrep wrapper that missed matches there).
14. **Cross-check bandwidths against computed traffic**: GB/s x call time = bytes per call (head 512 at 8K: 9.0 GB measured vs 9.1 GB for one K/V read per workgroup). `GPU L3: Busy` ~100% is not saturation (micro pulled 1.8x the bandwidth at the same Busy). Rising `L3 Miss Ratio` with DRAM read far under peak = latency, not bandwidth.

## Standalone harness (no product build)

`test/sdpa_ocl_xe_hpg/s7a/perf/opt/` (untracked): `sdpa_bench` (arm `name=src:file.cl[:options]` / `name=bin:file:micro`, `--verify --rows N`/`--all-rows`, `--shape seq,heads,kvheads,head`), `assemble.py` (product `.cl` + JIT header → runnable kernel), `mkvar.py`/`tile.py`/`sweep.py`/`reparam.py`, `vt*.sh`, `isa.sh`/`isasum.py`, `mb/`. Traps: a `#define` in a variant source overrides the harness `-D` (`#ifndef` guard), `--rows` is sampling with replacement (exhaustive is `--all-rows`), a wrong kernel also benchmarks fast (look at the verify line), micro.bin is shape-specific.

Decode (2026-10-28, `test/g4dec/`): `probe_g4` (the lfm2 decode probe + `PROBE_FILTER`, `PROBE_OPTS`/`_BYCHAN_OPTS`/`_I8TOK_OPTS`/`_F16_OPTS`, `PROBE_TWIN`, `PROBE_KDIR`, `PROBE_DUMP`, `PROBE_BENCH=<n>` with a 256 MB fill per launch and a spill readout) matched cliloader within ~5% at 33K keys on gemma-4; at short context the real KV is L2-resident, so confirm short prompts on the model. Per-decode-token kernel time on a real model: `cli_pair.sh` (`-ic 1` and `-ic 129` under cliloader) + `dec_an.py` (subtracts per family, `RUNS=` for multi-prompt files). E2E mirroring the daily command: `e2e_daily.sh` / `e2e_model.sh` (`EXTRA=--disable_prompt_permutation`) + `e2e_an.py`; offline ISA identity: `isa_ab.sh`.

## A770 (DG2) ceilings (MEASURED)

dpas.8x8 f16: 1 in-flight per thread (~30 ns), saturates at ≥4 threads per EU (128 GRF 104 TFLOPS, 256 GRF 116). dpasw = dpas speed. L1-hit block read 64 B 2.84 msg/clk, 256 B 0.93 msg/clk, L1 latency ~137 ns, L3 ~4.1 TB/s, SLM 12.4 TB/s, GDDR 2–3 TB/s. 256 GRF = 4 threads/EU, 128 GRF = 8. Occupancy halves at SLM 47.7 KB/WG.
