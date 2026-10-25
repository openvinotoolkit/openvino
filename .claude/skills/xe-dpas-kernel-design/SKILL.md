---
name: xe-dpas-kernel-design
description: Playbook for designing Intel Xe2 (SG16) and Xe-HPG/DG2 (SG8) OpenCL DPAS/XMX kernels - A/B/C operand mapping, VNNI, tiling across subgroups and workgroups (sg_per_wg, tile_keys, wgTQ, alpha nesting), 256GRF vs 128GRF, head-size generality. Use for sdpa_ocl, choose_config, DPAS tiling, SG8 port, "wrong output after tiling change".
---

# xe-dpas-kernel-design

Checklist for defining or changing the operand mapping and tiling config of a DPAS kernel.
In-depth evidence/tables/measurements are in `src/plugins/intel_gpu/docs/ocl_perf_guide/01-dpas-and-tiling.md` (hereafter "chapter 1"). Memory/IO → chapter 02, numerics → chapter 03, spill/ISA → chapter 04, methodology → chapter 05.

Repository rules (AGENTS.md): builds/GPU tests/benchmarks are **run by the user** (present exact command blocks and receive the results). Do not claim a performance improvement without measurement. Attach the device (B580/B70/DG2) and source to every number and distinguish MEASURED/ASSUMED. Always check NaN/Inf, accumulator dtype, dynamic shape.

## 0. Check before starting (1 minute)
- Target arch: Xe2 (SG16, `short8` A, has 2D block IO) vs xe_hpg (SG8, `int8` A, no 2D block IO). Code location: `sdpa_ocl_config.cl:12-17` (`SG8`, `DPAS_A_T`).
- The source is the truth for the current state: `sdpa/sdpa_gen_ocl.cpp` `choose_config_kq_only`, `solve_sv_split`, `slm_bytes`, `tiling_fits_device`; kernels `sdpa_ocl.cl`, `sdpa_ocl_config.cl`; doc `docs/sdpa_ocl.md`.
- Knob names in memory notes may be stale (`SDPA_OCL_BLOCK_SKIP`, `_DKS_ACTIVE`, `_PA_CUR_GRAN/SIDE/SV_TRIM/V_PREFETCH` were removed by the 2026-09 refactor). grep before using.

## 1. DPAS operand rules (the starting point of every mapping)
- `float8 C = intel_sub_group_f16_f16_matrix_mad_k16(short8 A, int8 B, float8 C)` (`bf16_bf16` for bf16). M=8 rows (`DPAS_ROWS`), K=16 (`DPAS_K`), N=SG width. The accumulator is float.
- **lane = the column index of each matrix**: for A, lane=K element, component=row (M). For B, lane=column (N), 8 dwords = 16 K values in VNNI (2 consecutive K = 1 dword). For C, lane=N, component=M.
- B is the operand fixed to VNNI (no transpose flag). The input read as VNNI (V) is B; the other side (P) is A.
- SG8: A is `int8` (lane l = K 2l, 2l+1, low half = 2l), B dword j = K 2j, 2j+1, N=8. **Using the SG16 `short8` form on DG2 compiles without error, but DPAS disappears from the ISA, giving silent garbage** → compile offline with `ocloc -device dg2` and confirm the `dpas` count in the ISA is not 0.
- sdpa_ocl mapping: KQ = A:K (lane=head dim, component=key) × B:Q (lane=query) → C lane=query. S*V = A:P (lane=key, component=query; via SLM) × B:V (VNNI, lane=value column) → C lane=value, component=query.
  Benefit: softmax max/sum is an in-lane component reduce (no shuffle). Cost: S is transposed through SLM, alpha is redistributed per component with `sub_group_broadcast`.
- micro uses A=Q/B=K in KQ (transposed), A=V/B=S in VS. Same result (verify_micro_kq_dpas maxerr=0). micro's in-place VNNI stride dequant cannot be imitated in OpenCL (IGC floor).
- decode (q=1): A=Q (M=Q_PER_WG ∈ {1,2,4,8} = GQA heads), B=K (lane=key). M is the DPAS repeat count, so only 1/2/4/8.
- Applying the same permutation to A and B on the contraction axis (depth) leaves the result unchanged → a layout mismatch (u4 adjacent nibbles) is solved with one permutation in Q staging.

## 2. Understanding the tiling structure
- KQ: subgroup tile `kq_sg_tile_keys(16|32) × kq_sg_tile_queries`, WG array `pwk × pwq`. wgTK = tile_keys*pwk, wgTQ = tile_queries*pwq, `sg_per_wg = pwk*pwq`. S*V re-partitions the same subgroups as `(sv_sg_per_wg_scores × sv_sg_per_wg_values)`.
- Dispatch: `LWS={SG, sg_per_wg, 1}`, one WG per head, PA query-block stride = the jitted wgTQ (host/jit/dispatch derive it together through `make_problem`).
- The key axis is a reduction: pwk subgroups merge via SLM atomic max + `S_sum_slm` → the larger the subgroup/k0 tile, the more merge·barrier overhead.
- SLM = `D_MAX*wgTQ*2 + wgTK*wgTQ*2 + wgTQ*pwk*4 + wgTQ*4` (default h128 = 17536 B). Xe2 128KiB / xe_hpg 64KiB, WG ≤ 1024 work-items — **≤ 512 at 256GRF** (16 x 4 SG16 subgroups at 256GRF = `CL_INVALID_WORK_GROUP_SIZE`, B70 MEASURED; `tiling_fits_device` does not know this).

## 3. Invariants that silently produce wrong output (check for every config candidate)
1. `sv_tile_values * sv_per_wg_values >= vd_max`. 2. `sv_tile_scores * sv_per_wg_scores == wgTQ`.
3. `sv_per_wg_values * sv_per_wg_scores == sg_per_wg` — breaks if you change only `kq_sg_per_wg_keys` (timings from that state are invalid).
4. **alpha[] nesting**: for every subgroup, `sg_i0_sv >= sg_j0_kq` and `sg_i0_sv + sv_tile_scores - 1 < sg_j0_kq + kq_tile_queries`. No kernel `#error` → only the per-subgroup loop in `solve_sv_split` protects it. Do not look only at tile sizes.
5. Others: `kq_sg_tile_keys ∈ {16,32}`, cp block = DPAS_K = page size, u4 `DKS_ACTIVE` even, Q staging is round-robin (1:1 assignment gives wrong results at head ≥ 256 where `q_blocks*DKS > sg_per_wg`).
- The overrides `SDPA_OCL_KQ_TILE_KEYS/_TILE_QUERIES/_PER_WG_KEYS/_PER_WG_QUERIES` re-derive the S*V split and assert (sweep without rebuild). Confirm they took effect with `SDPA_OCL_TRACE_CONFIG=1`. `setupvars.sh` wipes positional arguments, so capture the env before sourcing.
- **tile_keys=32 + pwk≥4 + query_blocks≥2 gives wrong results of unidentified cause** (does not reproduce on the current MICRO_MATH=1 path). Do not use it in a shipped config.

## 4. Config selection procedure
1. Write the A/B/C (lane, component) table and put the reduction axis on the component side (chapter 1 §2).
2. Put the invariant solver on the host and make overrides pass through the same solver (§3).
3. Filter candidates with a resource model: SLM formula, WG size, live registers (number of float8 C tiles + temporaries). ocloc spill predictions are **for reference only** — all 8 candidates were predicted spill 0, yet 4 spilled 3.3-19KB at runtime. The final verdict is runtime cliloader `SPILL=`.
4. Rebuild-free env knobs + trace first. Then ISA A/B by editing `#define`s in the dumped source + `ocloc` (chapter 04).
5. Candidate grid: total subgroup count (`aligned_q/wgTQ × heads × sg_per_wg`), wgTK (64/128), wgTQ (32/64), sg_per_wg (8/16), **GRF (128/256) paired with the tile**.
6. Timing only after the accuracy suite passes. Fixed clock (B70 2800MHz), per-kernel device time, 3 or more passes, representative shapes + synthetic in parallel.
7. Do not generalize across shapes: do not turn one shape's result into a law; find the discriminator (e.g. `V_TILES`). Before changing a default, check for regressions in prefill/mixed/decode and other heads·dtypes·archs.

## 5. Rules confirmed by measurement (device/source in chapter 1)
- **PA PREFILL per head size (2026-10-23, chapter 1 §6.6)**: mirror the reference's **workgroup** tile, not only its subgroup tile — K/V is re-read `seq / wgTQ` times. Shipped on Xe2 with 256GRF + the pipelined loop: d_max 64 (16,64,8,2), 128 (16,64,16,2), 256 (16,32,8,4) = micro's WGs; 512 (16,16,8,4) = 128 x 64 (micro's 256 x 64 needs tile_keys 32). With `MICRO_MATH`, only micro's key tile gives micro-identical output. A constant kernel gap grows with prompt length in e2e → measure prefill at 2K..32K.
- **GRF**: Xe2 default is 128 (PA PREFILL at d_max 64-512 uses 256 since 2026-10-23). `SDPA_OCL_256GRF=1` halves the thread count, making the default tile +25% slower (B70 llama-3.1-8b) and +47% slower at head 72. It only pays off **when paired** with a large tile blocked by spill (sv_sg_tile_scores ≥ 64) (256GRF + tq32/pwk4/pwq2 = 1,794,065 ns vs micro 1,930,454, B70 f16 q=4096). xe_hpg always uses 256GRF.
- **occupancy% (cliloader) can be inversely correlated with speed** → do not use as a metric. "Static message count / loads per unit of work" alone is also a wrong metric (gpt-oss: 4096 total subgroups was optimal, 2048 lost).
- **The WG-size axis and the k0-tile axis are independent** and shape-dependent: key-reduction overhead (large sg_per_wg, large wgTK = slow) vs thread starvation (small sg_per_wg = slow). On PA prefill q=1024 head128, sg16/wgTQ64/wgTK64 (A') beats sg8, sg4 (2.27M vs 2.80M/3.89M ns, B70).
- **MIXED (cache+current)**: narrow the key tile (wgK 64) and keep the threads (sg16). Because of the WG barrier, everyone pays the slower side's cost on the boundary k0 tile. Reducing sg collapses performance (372k vs 146k ns).
- **decode**: `SG_PER_WG = (v_head/16 >= 16) ? 16 : 8` (V_TILES discriminator), M is limited by a `live_grf_estimate` budget of 112 (M=8 head512 = spill 34432, 3.14x). TLP-bound sign: 2.5x the work but only +0.5% time.
- **head size**: `D_MAX` = rounded up to a power of 2. head 72 reduces only the depth loop with `DKS_ACTIVE=ceil(d/16)` (the S*V split keeps D_MAX). The cause of the 8.11x gap at 72 was not the tile but the block2d gate (`%64`→`%16` + base fixup) (6.96x) — **check that the IO path is enabled before suspecting tiling**.
- k≠v head: re-derive the S*V split with `vd_max`; if that fails, Tier 2 (KQ 16x16, pwk4 pwq4).

## 6. xe_hpg (SG8) porting check
- There are 3 places that must be rewritten: the K A-operand load (`k_tile_dword`: `block_read(uint)` per row, guard `dword_ok`+whole_tile, unaligned/tail falls back to aligned dwords), the S*V pA read (`as_int8(block_read8(local uint*))`), and the DPAS A type. softmax etc. are shared because lane=query.
- No 2D block IO/prefetch → must be turned off with a JIT switch (`#ifdef` is not possible). global `block_read uint` gives wrong results at odd pitch/2B offsets. K values beyond `d` and beyond `k` are 0 (guards against 0*NaN=NaN).
- Seed tile KQ 16x16, sg 4x2, 256GRF (DG2 microbench: sg4x2 3353 µs min, 128GRF spills 3.8-11KB in every config; MEASURED DG2 S2). Real-model DG2 performance is not in this repository (DG2 PC) — do not claim it.
- The `SDPA_OCL_NEG_SG8=1..6` negative controls must each **make** the sharp-softmax test **fail**. `add_stage` swallows codegen failures and silently demotes to opt → confirm with a census that an sdpa_ocl dispatch actually exists.

## 7. Common mistakes
- Trusting the "fast" timing of a config that violates an invariant (accuracy first).
- Not checking whether the env actually took effect (if the same SLM/GWS/LWS comes out, the knob did not take). Check that the path is live (e.g. `ATTENTION_BACKEND=PA`).
- Shipping on the basis of ocloc spill=0. Checking cliloader `SPILL=` is mandatory.
- Attributing from a single control (bidirectional on/off needed).
- Reusing B580 numbers for B70/DG2.

## 8. What to include in the deliverable
Table of the chosen config (tile, sg_per_wg, SLM, GRF, spill), results of checking invariants 1-4, accuracy suite results, device/clock/methodology, failed candidates and the reasons (chapter 1 §13 format), MEASURED/ASSUMED distinction.
