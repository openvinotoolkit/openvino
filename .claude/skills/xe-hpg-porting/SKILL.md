---
name: xe-hpg-porting
description: Playbook for porting or adapting an Xe2 DPAS/2D-block OpenCL kernel (sdpa_ocl) to Xe-HPG (DG2/Arc-A, ARL-H, SG8, no 2D block IO) in OpenVINO intel_gpu - SG8 vs SG16, HpgTier mask, micro-lane fallback, gates and landmines. Use when touching xe_hpg, DG2, ARL-H, SG8, TEST_USE_SDPA_OCL_HPG, tiers, or sdpa_micro fallback.
---

# Xe-HPG (DG2/ARL-H) porting playbook

Depth: `src/plugins/intel_gpu/docs/ocl_perf_guide/05-methodology-and-pitfalls.md` chapter 5.7 (step table, DG2 facts, tier table), design doc `src/plugins/intel_gpu/docs/sdpa_ocl.md` "xe_hpg bring-up" section (:62).
Methodology (A/B, attribution, performance discipline) is in the skill `ocl-kernel-ab-methodology`. Kernel techniques are in chapters 01~04.
Working copies/plan: memory `sdpa-ocl-xe-hpg-plan` (hub), `-facts`, `-s7-performance` (top priority), `test/sdpa_ocl_xe_hpg/`.

## 0. Current state (2026-10-06)
- S0~S6c done: plain SDPA f16/bf16/mask/causal/sink/dynamic/q<=1 and plain i8 KV work on the xe_hpg SG8 arm. In committed HEAD `d20cac7`, `kHpgTiersReady` is `PLAIN_F16_STATIC | PLAIN_EXT | PLAIN_I8`.
- The uncommitted working tree has the `PA_PREFILL` bit, separate PREFILL/MIXED routing, and a `KV_TILED` preprocessing prototype. This change differs from the vISA raw-kernel policy that passed the standalone gate, and it has not been validated by a product build, accuracy or performance. Even if the ready bit is visible in the current source, do not classify S7a as product-ready. Product integration of the raw kernel is separate follow-up work.
- **Until S9, xe_hpg default behaviour is unchanged**: the sdpa_ocl lane only when `TEST_USE_SDPA_OCL_HPG=1`. Rejected ops are routed to the sdpa_micro lane by the 2 `TEMP(S9)` sites (`sdpa/sdpa_opt.cpp:86`, `sdpa/paged_attention_opt.cpp:1623`).
- Real-hardware validation is **DG2 (Arc A770) only**. ARL-H (12.74) is included in the S9 defaults/cache tag but its **runtime accuracy/performance is unverified**. B70/Xe2 regression + corpus/pset invariance proofs are not waived for DG2-only work.
- The initial S7a PA PREFILL product path was confirmed by device-USM ABBA to be about 24–28x slower than micro at head128/32:8/seq4096. The follow-up standalone raw K/V kernel passed the 3% gate specified on 2026-10-04 on the measurement matrix, but that is not evidence for product integration/end-to-end/S6/SG16/B70/ARL-H. The detailed causes and results are in [performance case study §6.3](../../../src/plugins/intel_gpu/docs/ocl_perf_guide/06-performance-gap-investigation.md).
- Performance priority after S7: PA PREFILL > PA MIXED > plain q=1, per-issue FIXED/REFUTED/approved ACCEPTED·DEFERRED. Do not generalize DG2 measurements to other Xe devices or unmeasured shapes.

## 1. Hardware difference table (MEASURED = DG2 S0/S2 real hardware)
| Item | Xe2 (B70) | Xe-HPG (DG2) |
|---|---|---|
| subgroup | SG16 | **SG8** (`get_subgroup_size(arch)` already 8 in sdpa_gen_ocl.cpp) |
| DPAS A operand | `short8` (lane l=K(l)) | **`int8`**, lane l = K(2l, 2l+1) in one dword (H1 confirmed 8/8, permutation control 24/24) |
| DPAS B / C | VNNI B, lane=column | B `int8` dword j=K(2j,2j+1), lane n=column; C lane=column, component m=row |
| 2D block IO (read/write/prefetch) | yes | **no** (compile rejected, extension not advertised). The pragma only warns + macro undefined -> turn it off with a jit switch, not `#ifdef` |
| GRF | 64 B (128GRF = 8KB) | 32 B (128GRF = 4KB): Xe2 tiles spill. Start with `-cl-intel-256-GRF-per-thread` as mandatory |
| SLM / WG | 128 KB | **64 KB**, max WG 1024 (the host had no check -> `tiling_fits_device`) |
| local block IO | yes | `local uint/ushort` block read/write OK without a pragma |
| split-matrix MAD | yes | DG2 only (not on ARL-H) -> do not use it as a required path |
| decode (PA GENERATE) | `sdpa_ocl_decode` (SG16, 2D block, block_size==SG) | originally `pa_single_token/pa_gqa`, not micro; SG8 `sdpa_ocl_decode` is out of scope (keep opt, xe2+ reader gate) |
- **SG16 `short8` DPAS on DG2 compiles without error and the ISA has no DPAS** (the input loads disappear too) -> silent garbage. Check the `dpas` count on every SG8 change (fail if the `dpas` column of the `sdpa_ocl_ab.py hpg` TSV is 0).
- DG2 `ocloc -device dg2`, ARL-H `-device arl-h` (offline, optional, approval before running). The general levels of `sdpa_ocl_ab.py` are fixed to `-device bmg`; only the `hpg` subcommand takes the device as a parameter.

## 2. Porting structure (settled in S2~S6)
- Shared kernel + **`#if SG8`** (`SUBGROUP_SIZE == 8` in `sdpa_ocl_config.cl`). The Xe2 path must be byte-identical (S4: SG16-neutral refactor counting keys in **`DPAS_K` units** instead of `SUBGROUP_SIZE` -> L1 same 2491 + modes 8937, ISA SAME 92+162).
- Layout-dependent rewrites are **3 places** for plain f16: (1) the K A-operand load in KQ (`k_tile_dword`: dword block read per row, tail/unaligned uses a 2-ushort fallback), (2) the pA read in S*V (`as_int8(intel_sub_group_block_read8((local uint*)&S_slm[(cp*WQ+q0)*8]))`), (3) the A type of the DPAS call. softmax state/alpha/O store/mask/sink/bidir/scale/causal/V gather are lane=query/value, so they are shared with SG16 (no porting needed).
  S_slm write: `vstore4(uint4)` at half offset `(key_block16*WQ+query)*16+(key%16)`. alpha uses `intel_sub_group_shuffle`.
- K direction choice (D=128 DG2 mini bench, unit us): K1 scalar gather 365 (adopted), K2 vload8+pack 370, K3 SLM transpose 449, K4 A=Q/B=K 344 (KQ only; not adopted), K0 contiguous 286.
  V0 = lane=value gather. Tail rule: dword 0 if `db*16+2*idx >= D`, guard **only one of K or Q**.
- Tile/GRF: **256GRF, KQ 16x16, 4 subgroups along the key direction (`sg4x2`), spill 0** (S2 small prototype). 128GRF spills 3.8~11KB in every configuration; k16q32/k32q32 spill even with 256GRF. Limitation: this is not a proof of optimality for the whole product PA. `get_build_options` **forces** 256GRF on HPG, so `SDPA_OCL_256GRF=0/1` is not a 128/256 experiment.
- Places that can be silently wrong on SG8 (blocked with SG8 `#error`, `sdpa_ocl_config.cl`): all 2D block IO paths, the K/V readers of compressed/PA MIXED, `pa_k_comp_by_token` (rereads tokens 0..7), the scale in `pa_v_comp_fold`/`v_i8_comp_fold` (2 keys per lane vs 1 half), the half16 width of `mask_tile_2d`, places where S_slm assumes `SUBGROUP_SIZE==DPAS_K==page(16)`.
- `v_tile_b2d16` (`#if USE_2D_BLOCK_IO_KV || IS_PA_MIXED`) unconditionally uses a 2D transform builtin in PA MIXED f16 -> DG2 compile error even with `USE_2D_BLOCK_IO_*=0` (S7b must replace it with a scalar Kc/Vc reader).
- UB gate: `block_read uint` with an odd row pitch gives wrong results (err 49), `block_read_us` at a 2 B offset is wrong too -> contiguous readers only with **even pitch + 4 B alignment** (runtime `k_dword_ok`), otherwise fallback.
- d-major BY_CHANNEL K page: cannot be read by design -> **DM reader** (S8b) first (DM-first). u4 nibble order is confirmed against the writer in S8c.

## 3. Host: gates, tier mask, routing
- Predicate split (S3): `sdpa_ocl_arch_ok` / `sdpa_ocl_hpg_enabled` / `sdpa_ocl_decode_reader_available`. The old single predicate `sdpa_ocl_selected = env && immad && arch>=xe2` served several meanings.
- **landmine**: `by_channel_token_major_readable()` (paged_attention_opt.cpp) uses `sdpa_ocl_selected` as a stand-in for "a decode reader exists", so widening only the predicate creates token-major pages on xe_hpg and the d-major GENERATE reader throws.
- **Routing trap**: `sdpa_ocl_selected` true but `supported()` false -> `none` (opt kernel). It does not fall to the micro tail (`paged_attention_opt.cpp:1615-1620`). `add_stage` swallows codegen exceptions, so an SG8 build failure is **silently demoted to opt** -> confirm the stage exists with a census.
- Tier bits (`sdpa_ocl_hpg.hpp:20-29`): in committed HEAD only PLAIN_F16_STATIC, PLAIN_EXT, PLAIN_I8 are READY. PA_PREFILL/PA_MIXED_F16/PA_FEATURES/PA_I8_TOKEN/PA_I8_CHANNEL/PA_U4 have not completed the product gate. The PREFILL-only routing and the `KV_TILED` pre-pass in the current working tree are an unverified prototype, and the raw-assembly measurement results without a separate pre-pass are not connected to the product. Re-check the per-tier step status in [performance case study §6.3](../../../src/plugins/intel_gpu/docs/ocl_perf_guide/06-performance-gap-investigation.md).
- When enabling a new tier bit: add the bit to `kHpgTiersReady` + settle the `hpg_tier_required` boundary + add per-op expectations to the test mirror `expected_dpas_backend_for(...)` (while at tier 0, mirror with the **exact expectation**, not "SKIP") + align the `sdpa_ocl_serves_prefill_mixed()` guard.
- The `SDPA_OCL_*_2D/_1D/PA_CUR_F16` envs are **ignored** on xe_hpg by `block2d_io_allowed(sg)` and forced to 0. Tiling uses `tiling_fits_device()` (SLM 64KiB, WG 1024), shared by `choose_config` and `supported()` (KQ overrides are evaluated too).
- The model cache descriptor (`compiled_model.cpp:282-303`) has no lane/stage layout/`TEST_USE_SDPA_OCL`, so old micro-lane blobs get reused -> an S9 cache tag is needed (both xe_hpg DG2 + ARL-H; the cache strings for Xe2/other archs/no-XMX/builds without oneDNN stay unchanged).
- The ARL-H static decode exclusion rule (`sdpa_opt.cpp` `is_ARL_H = gfx 12.74`) is kept independently of the default lane switch. Identification is by `gfx_ver` (`device.cpp` puts 12.74 on a separate row from DG2 12.55~12.57).
- Micro-lane limits on the host are as upstream: k==v, runtime mask with 1 element not supported, unaligned single-token not supported, f32 Q/O rejected, `micro MIXED + bidir tti` unsupported (opt ignores the mask).

## 4. Verification procedure
1. **What is possible on B70**: host routing/jit generation/fake-device unit tests, byte-identity proofs for SG16-neutral changes, offline compilation of fake-arch dumps:
   ```bash
   OV_GPU_ARCH_OVERRIDE=xe_hpg TEST_USE_SDPA_OCL_HPG=1 SDPA_OCL_HPG_TIERS=all \
     bash test/sdpa_ocl_gtests.sh dump <tag> U1 U3 F2     # ENABLE_DEBUG_CAPS build (+F1, U2)
   python3 test/sdpa_ocl_ab.py corpus <dir> --base <snapshot>
   python3 test/sdpa_ocl_ab.py hpg --corpus <dir> --base <snapshot> --device dg2 --grf256 [--define SDPA_OCL_SG8_ARM_READY] --out <dir>
   ```
   TSV: rc, first_error, sg, dpas, dpas_forms, simd, numGRF, spill(B), slm, inst, any_2d, any_1d. **The `dpas` column must not be 0 for any in-tier kernel.** Confirm 2D leaks with `any_2d` + the `ARM_READY` mutation pass2 (looking only at first_error hides them). **Do not interpret gtest results run under a fake arch; only the dumped sources are meaningful.**
2. **DG2 real hardware is mandatory**: SG8 accuracy, census, performance. Prior approval for every run (who runs it, workload, budget, arm/order/warmup/repetitions). New kits/logs go under `test/sdpa_ocl_xe_hpg/`.
3. Pair every new SG8 mapping with a **sharp-softmax negative control**: `SDPA_OCL_NEG_SG8=1..3` (K pair swap / S*V A transpose / S_slm pair swap) must FAIL, `=4` (force the unaligned-K fallback) must PASS, and `=5/6` (mask width/lane offset) are controls too. For the KVC family, **perturb the reference** with `SDPA_KVC_NEG=pair_swap|scale_alias|zp_off` (all FAIL). Groups are `HPG_SDPA, NEG_sg8_*, KVC*` in `test/sdpa_ocl_gtests.sh`.
4. **The verdict is the census**: while the tier mask rejects, "PASS with HPG=1" is not sdpa_ocl evidence. Confirm >=1 `sdpa_ocl` line in `Enqueue stage` from `OV_VERBOSE=4` (S5 DG2: U1 FAIL 0/PASS 328/SKIP 109, yet sdpa_ocl dispatch was 0). Reproduce a single gtest with `TEST_USE_SDPA_OCL_HPG=1 ./bin/intel64/Debug/ov_gpu_unit_tests --device_suffix=1 --gtest_filter=...`.
5. **Xe2 invariance**: new SG8 code must be A on the B70 corpus/pset/L1/L2 (or change only the intended configs). `gtests.sh diff` 14 groups identical, NEG_* FAIL count unchanged.
6. Attribution of existing DG2 failures uses **two controls, S1/dg2base + current micro/opt**: the 133 F1 cases are unrelated to sdpa_ocl (bf16 head 486/512 `CL_OUT_OF_RESOURCES` 128, bf16 runtime-scale 4, f16 runtime-scale 1 are **dependent on in-process execution order**, PASS standalone). Pin these names as the baseline failure list so that ops falling to opt after S9 do not look like new regressions.
7. **DG2 micro-only pass list** (all must PASS on the ocl lane before S9 for it not to be a regression): U1 24 = `micro_sdpa_prefill` 13 + `u4_mixed_micro` 5 + `update_shape` 1 + `sink/{0,1,2,3,6}` 5. Still SKIP: `micro_sdpa_prefetch_k` (below xe_hpc), 6 `token_type_micro_sdpa_mixed`, `qq_bias_token_major`, runtime-scale (`arch<xe2`).
8. Mutation verification: deliberately break a gate/mirror and confirm only the predicted tests FAIL (mutations are **standalone builds**; use `&& false` instead of removing variables).

## 5. Performance check loop (S7+)
Accuracy/actual dispatch -> static check (existing corpus, tile/SLM/GRF/private-array lifetime/duplicate gathers/barriers) -> short screening on approved representative cases -> if triggered, one-change controlled experiment -> §4.5 regression + the user's disposition.
- For a large gap, first check comparison validity in the order of [ocl-kernel-performance-investigation](../ocl-kernel-performance-investigation/SKILL.md), and tie VTune/GTPin/ISA each to one question. In S7a, usm_host inputs, the micro wrapper before final fusion, and the K/V gather kind guessed from the source alone could all have led to wrong conclusions.
- The S7a raw-kernel PASS is a vISA inline-assembly standalone harness result. Do not report "sdpa_ocl replaced micro" until actual product dispatch/census, product accuracy, end-to-end, S6 and the SG16/B70 invariants are closed separately. The uncommitted `KV_TILED`/preprocessing branch is a prototype different from that gate, and the latest checkpoint recorded it as an earlier implementation not needed for the final raw-kernel path.
- Comparison axis: **current OCL vs the current actual micro/opt** (check the actual stage, not the name), same Release build/DG2/driver/inputs/cache. Earlier OCL checkpoints are a separate axis. The S1 Debug micro table is not a performance baseline.
- ocloc spill (plain h64 7,968 B / h128 12,128 B / h256 28,064 B) is a **statically reported value**, not DG2 runtime spill/time. Querying the runtime `CL_KERNEL_SPILL_MEM_SIZE_INTEL` is possible (S2 PASS) but it is not latency.
- Common SG8 tuning requires the corresponding accuracy/performance regression if it affects the completed S6 plain/q=1.
- Tools: `test/sdpa_perf_hpg.py` (plain f16 static only, cumulative primitive average, warmup included, integer us -> do not call it a median), existing PA gtests + CLIntercept (first run/cold mixed in, no steady-state claims). If needed, a minimal C++ PA repeated-measurement extension after approval.

## 6. Don'ts
- Do not **run** readers that are not ready by bypassing the tier mask (`SDPA_OCL_HPG_TIERS=all`) (dump only).
- Do not leave SG16 `short8` DPAS code as-is on SG8 (silent DPAS loss).
- Do not invent DG2 results/numbers you have not seen, and do not rerun completed steps by interpreting missing local material as an incomplete step (ask the user for the exact tag/path/build identifier).
- Do not guarantee ARL-H runtime accuracy/performance from DG2 results or `ocloc -device arl-h`.
- Out of scope: deleting micro code, SG8 `sdpa_ocl_decode`/new token-major paths, new performance floors without approval.
