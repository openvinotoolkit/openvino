---
name: xe-spill-isa-profiling
description: Playbook for diagnosing register spill, scratch/TPM, 256-GRF and occupancy problems in Intel Xe2/Xe-HPG OpenCL (DPAS) kernels by dumping and reading GEN ISA (IGC ShaderDump, ocloc splice, iga64) and cliloader SPILL=/TPM= lines. Use for "spill", "TPM", "scratch", "256 GRF", "ISA dump", "VTune", "ocloc A/B", "always_inline".
---

# Xe spill / ISA / profiling playbook

In-depth evidence and numbers: `src/plugins/intel_gpu/docs/ocl_perf_guide/04-spill-isa-profiling.md` (hereafter "ch4"). Tiles/DPAS are in chapter 01, memory I/O in 02, numerics in 03, methodology in 05.

## Invariant rules (repo AGENTS.md + the user's memory)
- Builds, gtests, cliloader runs and ab runs are executed **only by the user**. Present exact command blocks and receive the results. Do only read-only analysis (grep, reading sources/dumps) yourself.
- Do not claim a performance improvement without measurement. Write the hardware (B580/B70/DG2) and source for every number and distinguish MEASURED vs ASSUMED.
- Always check NaN/Inf, accumulator (f32), precision conversion order, dynamic shape (shape_info). A tile override can produce *silently wrong output* with no crash/spill.
- A new `.cl` requires rerunning cmake (GLOB is configure-time). Commit messages are one line.

## 1. Observe first (runtime ground truth)
```bash
# setupvars.sh wipes "$@", so save the arguments before source
cliloader -d -dv python <llm_bench>/benchmark.py -d GPU.1 -m <model> -n 1 -ic 4 -pf <jsonl> 2>&1 | grep <kernel>
```
- Kernel name line: `SIMD16 REG128|256 [SPILL=N] [TPM=N] SLM=N GWS[..] LWS[..]`. With spill 0 there is no `SPILL=`.
- **Judge spill/TPM only from the runtime line.** ocloc offline spill was accurate for head-72 (2688 B) but wrongly showed 0 for 4 tiling sweep cases and the kc/vc B/C configs.
- 2-GPU box: without `ov_gpu_func_tests --device_suffix=1` (B70), sdpa_opt runs on the iGPU and the comparison is invalid.
- If configs should differ but SLM/GWS/LWS/time are identical, the toggle was not applied (setupvars `set --`, a stateful path where the PA kernel does not appear).
- occupancy% is not a speed metric (cases where it was higher yet slower). cliloader `-d` is host-bound, so gains on small kernels do not show in e2e.
- VTune was used for real B70 GPU Hotspots analysis (details and limits: `04-spill-isa-profiling.md` §4.1.1). Confirmed cases: occupancy 99.0%, XVE active 47.8%/stalled 52.2%, 5.581 ms/2 instances; in a separate OCL-vs-micro comparison, SBID 59.1% vs 34.2%, barrier stall 10.6% vs 2.5%. One collection had correct source/DWARF mapping but 0 stall PCs, so source-line attribution was impossible.
- Before collecting, record the VTune, driver and Metrics Discovery versions and the GPU BDF. On B70, Metrics Discovery 1.13.545 failed to initialize, which was resolved with 1.16.190. Pin the OpenVINO workload and the VTune target GPU to the same BDF. To keep the JIT source, set `OV_GPU_DUMP_SOURCES_PATH` together with `OV_GPU_MAX_KERNELS_PER_BATCH=1`.
- GPU active/stalled, SBID and barrier stall are metrics that narrow down bottleneck hypotheses, not direct evidence of bandwidth or of a specific source line. Resolve addresses with the exact zebin archive of the VTune result. If there are no stall PCs or the address mapping is uncertain, do not claim ISA/source-level attribution. The verdict comes from before/after VTune under identical conditions + cliloader device time + correctness A/B.
- XE/i915 counter and source attach permissions depend on the system sysctl. If an admin setting is needed, allow it only for the minimum period per the security policy, then restore it. Do not equate a `GTPin` warning with a collection failure; confirm that GPU metric data was actually recorded in the result.

## 2. Distinguishing spill kinds

When investigating an unexplained performance gap as a whole, first consult [ocl-kernel-performance-investigation](../ocl-kernel-performance-investigation/SKILL.md) and [chapter 06](../../../src/plugins/intel_gpu/docs/ocl_perf_guide/06-performance-gap-investigation.md).

In an additional A770/DG2 case, the original OCL had XVE active/stalled 15.6/81.8%, occupancy 47.9%, XMX active 2.0%, runtime spill 7,872 B. micro occupancy was 49.7–49.8%, XMX active 24.4–24.5%. K was already a dword block read and V was gather/packing, so the per-reader hypotheses were corrected by checking the executed ISA. VTune/GTPin root-cause analysis and its limits are in chapter 06 §6.3. A low-spill kernel can have no event/kernel data without AMPLXE_MORE_GTPIN_OPTIONS=-allow_sregs 1 even though it exits 0, and there was a case where source-analysis failed because the ELF symbol size of the fused micro final binary was smaller than the native text. Confirm the GPU events and the exact zebin/PC mapping.

| Observation | Meaning | First response |
|---|---|---|
| `TPM>0`, spill 0, scratch store/load inside the hot loop | **private array with a runtime index** | Make the index an unroll constant, or use a select chain + `sub_group_broadcast` (alpha[] case 0.8%) |
| the same `TPM=N` at every head size, new after a driver upgrade | IGC 2.41 no longer promotes a **uniform** runtime index (kvup writer `TPM=576` = `[2][9]` halves x 16 lanes, two helpers' arrays merged into one `alloca`) | List the `alloca`s left in `*_optimized.ll`; full-range `unroll_for` + uniform `if`, select over all slots (PREFILL 322 -> 172 us). 256 GRF / IGC knobs do not help; a `half16` + runtime subscript avoids scratch but was slower (ch4 §4.9-A) |
| `SPILL>0` | register pressure (N independent gathers live at once, etc.) | Improve the load geometry first, then the tile/GRF |
| `grep spill` matches in `.asm` | **meaningless**: `-abortOnSpill` in `//.full_options` | Read `//.spill size`, `//.private memory size`, `.zeinfo` |

## 3. Getting the ISA (without a rebuild)
1. Runtime dump: `OV_GPU_DUMP_SOURCES_PATH=./ OV_GPU_MAX_KERNELS_PER_BATCH=1` (jit prelude + standalone .cl) and `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir>` (.asm/.ll/.zeinfo). With `MAX_KERNELS_PER_BATCH=1`, 1 entry per dump. File names are hashes, so identify them with `grep -l 'kernel sdpa_ocl__prefill'`. Scripts: `test/dump_isa*.sh` (`dump_isa.sh` has an old hard-coded ROOT).
2. ocloc: `ocloc compile -file X.cl -device bmg -options '-cl-mad-enable -cl-std=CL3.0 -D...x4' -internal_options '...'` (`-device bmg`; `xe2` is interpreted as the iGPU. `-cl-mad-enable` goes inside `-options`). The original options are in the dump's `*_options.txt`/`*_internal_options.txt`.
3. splice A/B: dump prelude + `#include` of the working-tree `.cl`. Remove the `#define`s to sweep from the prelude with `sed`, then supply them again. Examples: `test/splice_head72.sh`, `isa_ab_mixed.sh`, `sdpa_ocl_ab.py` (L0/L1/L1'/L2/pset/built/hpg). kvup: `test/kvB/{splice.sh,isa.sh,modes_ab.sh,heads.sh}` (`kvup_splice.sh` / `isa_ab_kvup.sh` lost their dumps). An IGC ShaderDump `.cl` can hold several kernels of one batch (`//====` separators, e.g. kvup with the `paged_attention_opt` kernels): splice only the target's jit section.
4. **Prove reproduction first**: does a splice of the HEAD source match the runtime values (instCount/spill/dpas/hash) (`kvup_splice.sh check`, head-72 spill 2688 matched)? Compare absolute instCount/spill only between rows of the same table.
5. Disassembly: for a zebin, find the `.text.<kernel>` offset with `readelf -S -W` -> `dd` -> `iga64 -p=xe2 -d`. `/usr/bin/iga64` is available. micro's GEMM does not show up in the IGC dump (dummy `dpas.8x1 null`, `0xFADECAFE`), and compiling it with ocloc gives a fake ~709 instr -> never quote micro numbers from that.
6. After a build: confirm with `python3 test/sdpa_ocl_ab.py built --base <rev>` that the intended source is in the .inc (chunks joined)/.so/`ov_gpu_unit_tests` (the unit tests are statically linked and can be stale).

## 4. ISA reading cheat sheet
| Signature | Meaning / response |
|---|---|
| many `load.ugm.d8u32/d16u32 (1\|M0)` | SIMD-1 scalar gather (15/16 lanes wasted). For per-key scale/zp, use a lane=key wide load + broadcast (128 -> 0, B580) |
| dozens of `load.ugm.d8u32 (16\|M0)` | too many intermediate messages. Switch to `d32xKt` wide transposed / `load_block2d` |
| `load_block2d.ugm.*` | 2D block read (the target form; rules in chapter 02) |
| `dpas.8x8` count | steady-state count. IGC peels the first iteration (x2). 0 means DPAS was dropped (DG2 SG16 short8) |
| `dpas.8x1 null` + `0xFADECAFE` | micro blob dummy. Ignore |
| many strided `mov` (`<2>`, `<4;1,0>`) | int8 -> f16 widen/VNNI repack. The VNNI repack is IGC's floor; the byte convert chain is not (next row) |
| per element `mov :b <4;1,0>`, `mov <2>:w <- :b`, `mov <2>:hf <- :w`, `mov :uw <- <2;1,0>:uw` | `(half)(char)byte` / `(half)nibble` convert: 4-5 movs before `(q-zp)*sc`. Replace by the exact 16-bit-word widen `I8_TO_HALF`/`U4_TO_HALF` (`bfn`/`shr`/`xor` + `add`): i8 MIXED mov 5,147 -> 2,003 (ch03 §4.8) |
| 16 `mov` of a DPAS operand right after a branch merge | phi copies of a conditionally modified operand: write each row once on every path (AND with a broadcast mask instead of an `if`) |
| `load.slm.d64x64t` + 8 `mov (16) r<1>:d r<2;1,0>:d` | `block_read_ul4` reinterpreted as `int8`: the per-read deinterleave of ulong4 -> int8 DPAS operand (WIDE_READS Q_slm, +256 movs per key tile at d128). Wins only when memory-bound (ch02 §5.4) |
| `r[a0]` surge | lane-varying loop variable/runtime-indexed array (2 -> 274, 1.8x slower) |
| backward branch (target label on an earlier line) | a real rolled loop. A forward goto is a predicated skip |
| `goto/join` drop sharply, `d32x8t` decreases | the unroll was lost (guard-shape problem) |
| `sync.*` +-1..5 | SWSB jitter. Ignore if the send/dpas/math/load/store counts are unchanged |
| zeinfo `disable_mid_thread_preemption` true -> None | IGC ~600 instr threshold (<=598 true, >=601 absent). Not a functional change |
| `numGRF=256` / `-TotalGRFNum 256` | 256 GRF mode |

## 5. 256 GRF judgment (all MEASURED, ch4 §4.5)
- Only 256 on the default tile: **a loss** (llama-3.1-8b 2.18M -> 2.73M ns, B70).
- A large tile + 256 as a pair (tq32/pwk4/pwq2): 7% faster than micro (1.796M vs 1.930M ns, B70).
- Gather-bound kernels (head-72 scalar, u4 head-64): even when 256 brings spill to 0, +47%/+70% worse. Do not use spill bytes as the objective function; look at *why* the live set is large.
- xe_hpg (DG2) forces 256 (with 128, every measured tile spills). `SDPA_OCL_256GRF=0/1` alone cannot run an HPG 128/256 experiment.
- occupancy has 2 axes (WG count vs threads/WG). Cases where larger `sg_per_wg` and `kq_wg_tile_keys` were slower (head-128 prefill, 3.53M -> 6.86M ns, B70). The SLM-occupancy hypothesis was falsified by device measurement.

## 6. IGC pipeline traps
- Helpers use **only `SDPA_OCL_INLINE` = `__attribute__((always_inline)) inline`**. Plain `inline`, even when inlined, changes the *whole module's* pipeline and moves the ISA of unrelated configs too (121 of 122 graded C).
- Even with always_inline, the ISA moves on: widening parameter casts (split `size_t lane` + `int lane_i`), branching on a parameter that becomes a constant, merged coordinate arguments, **arrays inside the helper** (declare in the caller + `__private` pointer), loop-carried outputs (out-params).
- unroll: the trip count is a macro, the guard is at the **call site** (an `if(skip) return;` at the top of the helper loses the unroll, 2.3x slower; at the call site it stays 5.6 us). No lane-varying loop variables. Ablate by setting the loop bound to 0 (no `if(0)`).
- Private arrays: compile-time indices only. Under IGC 2.41 even a subgroup-uniform runtime index or loop bound over the array leaves it in scratch (`for (it < npo) a[it]`, `a[npo] = v`): loop over the full range with a uniform `if`, write `if (s == slot) a[s] = v` for every slot, read through a select chain.
- Branch intent in the source != IGC codegen (the causal block skip was flattened, +16 instr).

## 7. Removal levers summary (only ones with evidence)
select chain (removes TPM) / replace gathers with block reads (head-72 9.28x, spill 2688 -> 0, B70) / scale-zp hoist / shrink M or the tile (`live_grf_estimate`, budget 112, gemma-4 head-512 M=8 SPILL=34432, 3.14x slower) / 256GRF+tile pair / shrinking the tile (config A) vs shrinking threads (config C, 372,628 ns). **No evidence**: limiting with `#pragma unroll N`, reducing spill by splitting the kernel.

## 8. Verdict procedure
1. Prove "no unintended change" (identical instCount + opcode histogram) with an L2 ISA A/B (`isa_ab_*.sh`, `sdpa_ocl_ab.py l2`). If grade C, diff `beforeUnification -> afterUnification -> optimized -> ISA` with `compile_one(..., keep_dir=)` to find the first stage where they diverge.
2. Judge performance **only by device measurement**: A/B within one binary via env toggle, min/median, against the absolute value of the production baseline for the same shape. If the static instr count is the same or better but the time differs, ablate (cache sectors, dirty lines per thread: the store 4.8x case).
3. **Never attribute from a one-sided control**: an A/B must differ in exactly one thing (the head 486 case).
4. Small ratios (<~5%) are not predicted by microprobes (probe predicted -4.4% vs real-model parity). Large ratios (12%) agreed within 1 pp.
5. Record failed experiments too: int8 K 2D block 30.8 us > scalar 26.1 us, K_DIAG=2 observer effect (slower), half dequant slower, V cp-pair reuse no effect.

## 9. GPU hang / DG2
A `CL_OUT_OF_RESOURCES`/`clFinish` exception can be a hang-reset symptom: cross-check `dmesg | grep -i 'GPU HANG'` against the PID. DG2 ocloc spill (h64 7,968 B etc.; source `s6a/pass_real/results.tsv`, chapter 05 §5.7.2) is a compiler-reported value, not a DG2 runtime value. If the `dpas` column of `ab.py hpg` is 0, DPAS was dropped.

## 10. Report format
Tabulate changes/hypotheses/measurements: hardware, command, metric (cliloader SPILL/TPM/time or ocloc instCount), MEASURED/ASSUMED, regression risks, correctness items verified, open questions. If a build/run is needed, ask the user with exact command blocks.
