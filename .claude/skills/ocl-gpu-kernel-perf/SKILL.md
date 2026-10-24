---
name: ocl-gpu-kernel-perf
description: Entry point for high-performance Intel Xe OpenCL kernel work in OpenVINO intel_gpu (sdpa_ocl, DPAS/XMX, 2D block IO, prefetch, split barrier, softmax, int8/u4 dequant, register spill, VTune profiling). Routes to the specialised xe-* / ocl-* skills and the ocl_perf_guide docs. Use first when optimizing or debugging a GPU OpenCL kernel.
---

# Xe OpenCL high-performance kernels: entry skill

Full guide: `src/plugins/intel_gpu/docs/ocl_perf_guide/README.md`

## Symptom/task → skill

| Situation | Skill | Chapter |
|---|---|---|
| Deciding tiling/DPAS operands/SG·WG placement/GRF mode, wrong output after a tiling change | `xe-dpas-kernel-design` | 01 |
| Illegal/corrupted block2d reads, pitch/base rules, prefetch, split barrier, KV page layout | `xe-block-io-prefetch-barrier` | 02 |
| softmax/mask/sink, int8·u4 dequant (`0x6480`), scale/zp, accuracy regressions | `ocl-quant-softmax-numerics` | 03 |
| Large performance gap vs a reference, unexplained stall/low utilization, checking the comparison target/measurement validity | `ocl-kernel-performance-investigation` | 06, then 04/05 if needed |
| Ratio vs a reference drifting with prompt length/size, suspected clock throttling, prefetch that costs more than it saves | `ocl-kernel-performance-investigation`, `xe-profiling-tools-cookbook`, `xe-block-io-prefetch-barrier` | 06 §6.7, 07 §7.8, 02 §7.6 |
| spill/TPM, 256GRF, ISA dump, VTune source-analysis/GTPin, ocloc splice | `xe-spill-isa-profiling` | 04 |
| A/B, attribution, speedup claims, verification harness | `ocl-kernel-ab-methodology` | 05 |
| DG2/ARL-H/SG8 porting | `xe-hpg-porting` | 05 |
| cliloader/VTune/GTPin/ocloc commands·collection·interpretation, standalone harness, microbenchmarks | `xe-profiling-tools-cookbook` | 07 |
| IGC sinks loads/packing gets split into movs → inline vISA, verifier errors, wrong asm results | `xe-visa-inline-asm` | 08 |
| Starting a new kernel/performance planning/gate design, general design·experiment discipline | `ocl-kernel-dev-lessons` | 09 |

## Common principles (AGENTS.md and work discipline)

1. Numerical correctness takes priority over performance. Always check the impact on NaN/Inf, accumulator dtype, precision conversion, dynamic shape and shape inference.
2. Never claim a performance improvement without measurement. State the hardware (B580/B70/DG2), the methodology, and MEASURED/ASSUMED.
3. An A/B must differ in exactly one thing. Never attribute a cause from a one-sided control. Check the negative control first.
4. Static ISA metrics (instruction/message counts, spill predictions) do not guarantee speed. Runtime `SPILL=`/`TPM=` and device measurements are the final evidence.
5. Test data can hide errors (N(0,0.1), LCG power-of-two). First confirm the verification is sensitive, using a negative control/NaN-poison.
6. Execution such as builds, tests, benchmarks and dumps is done by the user by default (provide exact command blocks). Run things yourself only when the user explicitly allows direct execution in that session, and that permission does not carry over to other sessions/tasks. In the 2-GPU environment, check `ov_gpu_func_tests --device_suffix=1` (B70).
7. No scope expansion or opportunistic cleanup. Commit messages are one line.

## First 5-minute check

- Which hardware family is it (Xe2 SG16 / xe_hpg SG8)? If it differs, start with `xe-hpg-porting`.
- Confirm which kernel/device actually ran and that the inputs are in device memory. Also confirm the final native binary of the comparison arm.
- For long or heavy runs, confirm the clock actually reached (`act_freq`, `throttle/reasons`): a pin is a request, PL2 overrides it.
- Large performance gap with no known reason: investigate in the order runtime → profiler → ISA → matched A/B from `ocl-kernel-performance-investigation`. If a spill candidate shows up, apply `xe-spill-isa-profiling`.
- Wrong output: suspect in the order layout/gate (02), tiling invariant (01), mask type/OOB (02·03), and before blaming the consumer kernel, check the producer (e.g. dynamic_quantize) and a `TEST_USE_SDPA_OCL=0` control.
