# Intel Xe high-performance OpenCL (DPAS) kernel development guide

A collection of documents on the techniques, constraints, failed experiments and verification methods gathered over the whole development of `sdpa_ocl` (prefill / MIXED / decode, PA + plain SDPA, f16/bf16/i8/u4 KV-cache). The Xe2/B70 product performance investigation and the DG2 PA PREFILL standalone kernel gate results are in chapter 06, how to use the analysis tools in chapter 07, inline vISA in chapter 08, and general lessons to carry over to other kernels in chapter 09.
The sources are the project memory notes and session logs, the handoff/analysis documents and scripts under `test/`, and the kernel sources (`src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_ocl*.cl`).
Every number states, inside its chapter, the hardware (B580/B70/DG2) and whether it is MEASURED/ASSUMED. Where a note does not state the hardware, it is written as "B70 estimated".

## Chapters

| Chapter | Content | Matching skill |
|---|---|---|
| [01-dpas-and-tiling.md](01-dpas-and-tiling.md) | DPAS API, operand mapping (lane = column), SG/WG tiling, 4 tiling invariants, 256GRF, head-size generalisation, SG8 (xe_hpg) rewrite | [`xe-dpas-kernel-design`](../../../../../.claude/skills/xe-dpas-kernel-design/SKILL.md) |
| [02-memory-io-prefetch-barriers.md](02-memory-io-prefetch-barriers.md) | 2D block IO legality rules (width/pitch/base), transform/transpose constraints, base fixup, prefetch, split barrier, K token-major relayout, uc16 page read | [`xe-block-io-prefetch-barrier`](../../../../../.claude/skills/xe-block-io-prefetch-barrier/SKILL.md) |
| [03-numerics-softmax-quantization.md](03-numerics-softmax-quantization.md) | online softmax, mask/sink/causal, derivation of the `0x6480` XOR trick, u4 unpack, dequant identity, test-data pitfalls | [`ocl-quant-softmax-numerics`](../../../../../.claude/skills/ocl-quant-softmax-numerics/SKILL.md) |
| [04-spill-isa-profiling.md](04-spill-isa-profiling.md) | spill/TPM diagnosis, ISA dump, ocloc splice, iga64, `always_inline`, limits of static ISA | [`xe-spill-isa-profiling`](../../../../../.claude/skills/xe-spill-isa-profiling/SKILL.md) |
| [05-methodology-and-pitfalls.md](05-methodology-and-pitfalls.md) | A/B harness, attribution rules, performance-claim discipline, representative results table, timeline, 34 pitfalls, xe_hpg (S0-S9) | [`ocl-kernel-ab-methodology`](../../../../../.claude/skills/ocl-kernel-ab-methodology/SKILL.md), [`xe-hpg-porting`](../../../../../.claude/skills/xe-hpg-porting/SKILL.md) |
| [06-performance-gap-investigation.md](06-performance-gap-investigation.md) | the B70 9.28x/7% improvements, profiling/A/B/raw-kernel results of the initial DG2 24–28x gap, a branch-vs-reference decode gap (lfm2.5), a head 512 prefill ratio drifting with prompt length (PL2 clock + L3-miss knee), per-tool interpretation limits, and an investigation procedure reusable for other GPU kernels | [`ocl-kernel-performance-investigation`](../../../../../.claude/skills/ocl-kernel-performance-investigation/SKILL.md) |
| [07-profiling-tool-cookbook.md](07-profiling-tool-cookbook.md) | commands, interpretation limits, pitfalls actually hit and timing discipline for cliloader, VTune (overview/lsc-slm/source-analysis), GTPin, IGC dump/ocloc/iga64, cycle counters, microbenchmarks and standalone harnesses | [`xe-profiling-tools-cookbook`](../../../../../.claude/skills/xe-profiling-tools-cookbook/SKILL.md) |
| [08-inline-visa-asm.md](08-inline-visa-asm.md) | why inline vISA (`__asm__` + `.decl`/alias/`lsc_load`/`dpas`) was used, syntax, verified patterns, verifier constraints, the 16 things that broke, and the verification protocol | [`xe-visa-inline-asm`](../../../../../.claude/skills/xe-visa-inline-asm/SKILL.md) |
| [09-general-lessons-for-new-kernels.md](09-general-lessons-for-new-kernels.md) | bottleneck diagnosis decision tree, a sense of effect sizes, 10 design principles, accuracy and experiment-management discipline, tool pitfalls table, checklist for starting a new kernel | [`ocl-kernel-dev-lessons`](../../../../../.claude/skills/ocl-kernel-dev-lessons/SKILL.md) |

Entry skill: [`ocl-gpu-kernel-perf`](../../../../../.claude/skills/ocl-gpu-kernel-perf/SKILL.md) (technique → chapter map below).

## Technique → location map

| Technique | Chapter |
|---|---|
| DPAS (`intel_sub_group_*_matrix_mad_k16`), VNNI, SG8 vs SG16 | 01 |
| subgroup/workgroup block assignment (`sg_per_wg`, `kq_sg_tile_keys`, `wgTQ/wgTK`, `V_TILES`) | 01 |
| decode `SG_PER_WG` x M choice, wave model, rounding M up past the kv group | 01 (§8.3) |
| 256 GRF mode and tile size | 01, 04 |
| PA PREFILL tile per head size (mirror micro's workgroup), d_max 512 candidates, 256GRF work-group limit | 01 (§6.6) |
| long-prompt TTFT: prompt-length sweep, llm_bench MD5 output oracle, mixed-head cliloader parsing | 05 (§5.5.1, E17-E19), 07 |
| 2D block IO read/transform/transpose/prefetch | 02 |
| software prefetch (V_PREFETCH), split barrier | 02 |
| prefetch cost = cache-line requests (256 B line on B70), L3-only address prefetch (`PREFETCH_V_L3`), declaring `__builtin_IB_lsc_prefetch_*` in a kernel file | 02 (§7.6) |
| ratio drifting with problem size: clock throttling (PL2) vs a cache-size knee, fixed-clock length sweep, time / keys^2 | 05 (E20, §5.5.1), 06 (§6.7), 07 (§7.8) |
| VTune on B70 / xe driver (permissions, quoted arguments, bytes-per-call cross-check) | 07 (§7.3) |
| K d-major scalar gather → token-major relayout | 02 |
| block-reading a sub-64 B-row (d-major f16) page: wider-pitch surface + 32b transpose (`32r8x1c` layout probed), permuted contraction axis | 02 (§4.5) |
| exact widen with a non-integer zp (`U4_TO_HALF` / `I8_TO_HALF` on 16-bit words) | 03 (§4.8) |
| equal utilisation, more instructions: redundant work (causal triangle skip per subgroup) | 06 (§6.8), 09 (§9.3 #18-19) |
| softmax (exp2, alpha rescale, -inf/NaN guards), causal key-loop upper bound | 03 |
| int8↔f16 `0x6480 ^ byte` trick, u4 nibble, scale/zp separation | 03 |
| register spill, TPM, ISA analysis, ocloc splice | 04 |
| `TPM=` after a compiler upgrade (uniformly indexed private array in scratch, IGC 2.41), KV-cache writer store order across threads, cache-byte oracle | 02 (§10.3), 04 (§4.9-A), 05 (H), 06 (§6.9) |
| A/B proof levels, negative controls, layer-0 oracle | 05 |
| unexplained performance gap, cross-analysis of the real native kernel with VTune/GTPin/ISA, checking comparison arms for errors | 06 |
| step-by-step gap-closing ladder (24x → within 3% of micro), remaining defects at short lengths/small shapes/head-count | 06 |
| cliloader/VTune/GTPin/ocloc commands and collection modes, standalone harness, phase cycle counter, microbenchmark ceilings | 07 |
| inline vISA syntax, LSC/DPAS messages, verifier constraints, asm accuracy verification | 08 |
| bottleneck decision tree, design principles, fixed-policy gate, checklist for starting a new kernel | 09 |

## Before you read (honesty notes)

- VTune: there are session records using B70 GPU Hotspots and A770/DG2 GPU Hotspots/source-analysis. Counter collection success, native PC mapping and GTPin trace validity must each be confirmed. Measured examples and failed collections are recorded in [chapter 04 §4.1.1](04-spill-isa-profiling.md) and [chapter 06](06-performance-gap-investigation.md).
- Some commit hashes in the memory notes are no longer on the current branch because of rebases. Check the mapping table in chapter 05 first.
- The env toggles removed by the 2026-09 refactoring (`SDPA_OCL_BLOCK_SKIP`, `_DKS_ACTIVE`, `_PA_CUR_*`, `_MAX_BARRIER_V_PREFETCH`) remain only as historical description.
- Some contradictions between notes are left in the "uncertain" items at the end of each chapter (e.g. the number saying k16pwg4 beats micro conflicts with the tiling-constraints note and is unconfirmed).
- The DG2 S7a PA PREFILL standalone kernel comparison was measured. The 3% gate PASS is not proof of product integration, end-to-end, or all supported shapes. For the current boundaries see [chapter 05 §5.7](05-methodology-and-pitfalls.md) and [chapter 06](06-performance-gap-investigation.md).
- Line numbers refer to the 2026-10-06 working tree and will move soon. Search again by symbol name.
- `VERIFICATION.md` is a point-in-time record of a document/skill cross-check made before chapter 06 was added. Do not read it as a verification report for later results.
