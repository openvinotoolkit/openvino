# 01. Using DPAS/XMX and designing the tiling

Audience: what you need to know to hand-write DPAS (XMX) kernels in OpenCL C on Intel Xe2 (Arc B580/B70, SG16) and Xe-HPG (DG2, SG8).
Sources: the `sdpa_ocl` kernel family (`src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_ocl*.cl`, host `.../sdpa/sdpa_gen_ocl*.cpp`,
design doc `src/plugins/intel_gpu/docs/sdpa_ocl.md`) and the measurement records accumulated during development.

Notation
- **MEASURED**: a value measured on a real device (hardware + source given). **ASSUMED**: hypothesis/inference. No performance claim is made without a measurement.
- Hardware: **B580** = Arc B580 (Xe2), **B70** = Arc Pro B70 (Xe2, clock locked at 2800MHz, the current dev machine), **DG2** = Arc A770 (Xe-HPG, SG8); the iGPU has no XMX.
  Old absolute values from the B580 are treated as stale unless they were re-measured on the B70.
- Code citations are `path:line` (paths relative to `ocl_v2/`; where that is unambiguous, only the file name). Locations were re-checked with grep against HEAD on 2026-10-06; lines can move.
- Other chapters: memory/IO/barriers → `02-memory-io-prefetch-barriers.md`, numerics/quantization → `03-numerics-softmax-quantization.md`,
  spill/ISA → `04-spill-isa-profiling.md`, methodology/pitfalls → `05-methodology-and-pitfalls.md`.

---

## 1. DPAS built-ins and register layout

### 1.1 APIs used

| Use | Built-in (OpenCL, `cl_intel_subgroup_matrix_multiply_accumulate`) | Defined at |
|---|---|---|
| f16 | `intel_sub_group_f16_f16_matrix_mad_k16(A, B, acc)` | `sdpa_ocl_config.cl:36` (`DPAS_MAD_K16`) |
| bf16 | `intel_sub_group_bf16_bf16_matrix_mad_k16(A, B, acc)` | `sdpa_ocl_config.cl:22` |
| Others (unused) | i8/u8 `k32`, i4/u4 `k64`, split-matrix variants | Khronos spec; split-matrix is DG2-only, so no required path uses it |

pragma: `cl_intel_subgroup_matrix_multiply_accumulate`, `cl_intel_subgroup_2d_block_io`, `cl_intel_subgroups`, `cl_intel_subgroups_short`
(`sdpa_ocl.cl:1-6`). The build options add `-Dcl_intel_subgroup_matrix_multiply_accumulate -Dcl_intel_subgroup_split_matrix_multiply_accumulate`
(`sdpa_gen_ocl.cpp: get_build_options`, ~L1033).

### 1.2 Operation shape (SG16, Xe2): M=8, N=16, K=16

```c
// sdpa_ocl.cl:621 (KQ), :992 (S*V)
float8 C = intel_sub_group_f16_f16_matrix_mad_k16(short8 A, int8 B, float8 C);
```

| Operand | Type (SG16) | Meaning of the lane | Meaning of the component/register |
|---|---|---|---|
| A | `short8` | lane = 1 **K (contraction) index** (1 half) | component i = **row M=i** (fixed at 8 rows: `DPAS_ROWS=8`, `sdpa_gen_ocl.cpp:693`) |
| B | `int8` | lane = **column N** | 8 dwords = K=16 halves packed as **VNNI** (two consecutive K in one dword) |
| C | `float8` | lane = **column N** | component i = row M=i |

Key rule: **in every operand the lane is that matrix's "column index"** (for A, K is the column; for B and C, N is the column). Every mapping below follows from this rule.
- `DPAS_K = 16` is fixed for f16/bf16 k16 (`sdpa_gen_ocl.cpp:692`). `DPAS_ROWS` is the ISA repeat count, so only 1/2/4/8 are possible (decode uses it as M, §8).
- The accumulator is **float** (`float8`). Half/short accumulators are SG16-only and these kernels do not use them. The output is converted down only once, in the epilogue, by `ACC_TO_OUT8` (`convert_half8` /
  `_convert_bfloat168_as_ushort8`, `sdpa_ocl_config.cl:23,38`) → no precision loss during accumulation.
- **P (the softmax probabilities) is rounded to the input dtype (f16/bf16) before it enters S*V as A** (`PACK_SOFTMAX8`, `sdpa_ocl.cl:822`).
  The denominator `lsum`, in contrast, is the sum of the f32 `exp_tile` before rounding (`sdpa_ocl.cl:803-806`). This is a precision-conversion asymmetry; keep it in mind when comparing numerics (→ chapter 03).
- bf16: only the DPAS built-in name changes; the conversion functions (`DT_FROM_F32`, `MASK_TO_FLOAT*`) branch through per-dtype macros (`sdpa_ocl_config.cl:21-49`).

### 1.3 SG8 (Xe-HPG / DG2) shape

| | SG16 (Xe2) | SG8 (DG2) |
|---|---|---|
| A type | `short8` | **`int8`** (one dword per lane = 2 halves) |
| K mapping of the lane | lane l = K element `l` | lane l = K elements `2l, 2l+1` (low half = `2l`) |
| B | `int8`, lane = column | same shape, N=8, dword j = K `2j, 2j+1` |
| C | `float8` | `float8` (lane = column N=8) |
| Definition | `DPAS_A_T short8` | `DPAS_A_T int8` (`sdpa_ocl_config.cl:12-17`, `#define SG8 (SUBGROUP_SIZE == 8)`) |

- **Result of the wrong SG width (MEASURED, DG2 A770, `test/sdpa_ocl_xe_hpg/probe/S0_RESULTS.md`)**: compiling the SG16 `short8` A shape for DG2
  **compiles without error, but the ISA has no DPAS** (the input loads disappear too and only the stores remain; the optimized IR still contains `__builtin_IB_sub_group16_fdpas`).
  That is, **silent garbage output**. The initial prediction ("compile failure") was wrong. So `sdpa_ocl_config.cl` puts an `#error` at every point that would be silently wrong on SG8
  (`sdpa_ocl_config.cl:335-339` etc.; doc `sdpa_ocl.md` "xe_hpg bring-up").
  Detection: compile offline with `ocloc -device dg2` and check that the ISA's `dpas` count is not 0 (the `dpas` column of `sdpa_ocl_ab.py hpg`). (→ chapter 04)
- **SG8 DPAS mapping H1 confirmed (MEASURED, DG2 A770 IP12.55.8, `test/sdpa_ocl_xe_hpg/s2/S2_RESULTS.md`)**: 8/8 identical for f16/bf16 × M=1/2/4/8.
  A: lane l = K(2l,2l+1). B: dword j = K(2j,2j+1), lane n = column. C: lane = column, component m = row. The negative controls (NC-A/NC-B) matched the predicted permutations 24/24.
  The Khronos docs give no per-lane element table, so this was an "inference"; it was confirmed on real hardware.
- On SG8 a `float8` is 8 registers (GRF 32B) → the same tile needs 2x the floats per lane of SG16 (SG8: a 16×16 S tile = 4 DPAS, SG16: 2).

---

## 2. Operand mapping: which matrix becomes A/B

### 2.1 The sdpa_ocl (prefill / mixed) mapping

```
KQ  : S^T[key,query] = sum_d K[key][d] * Q[query][d]
      A = K  (lane = head dim d,   component = key)  <- a block read of K rows (row-major [key,head]) gives exactly this shape
      B = Q  (lane = query,        8 dwords = 16 d as VNNI)   <- block_read8 from Q_slm (staged in SLM as VNNI beforehand)
      C = S_tile float8 (lane = query, component = key)
SV  : O[query,v] = sum_key P[query][key] * V[key][v]
      A = P  (lane = key,          component = query)  <- 8-row block read from S_slm
      B = V  (lane = value column, VNNI)               <- 16b VNNI transform read of the token-major V page
      C = A_tile float8 (lane = value column, component = query)
```
Code: `sdpa_ocl.cl:542` (qB read), `:619-621` (KQ DPAS), `:912-920` (pA read), `:992` (SV DPAS). Doc: `sdpa_ocl.md` "Kernel flow".

**Why this mapping (design consequences)**
1. The KQ result has **lane = query** → the per-query softmax max/sum becomes a **scalar reduce over the 8 components inside a lane (+ the key blocks)**. No subgroup shuffle needed.
   Conversely, switching to A=Q/B=K transposes S (lane = key) and the max/sum becomes a `sub_group_reduce` (`pa-k-cache-layout-d-major` analysis; micro actually does it this way).
2. P's register layout (lane = query, component = key) is the **transpose** of what SV's A requires (lane = key, component = query), so P makes one trip through SLM
   (written with `vstore4`, read with `block_read8`; `sdpa_ocl.cl:822`, `:916`). This is why S_slm exists and is the core of the SLM use (§7).
3. SV's C is (lane = value column, component = query), so the online-softmax **alpha rescale** has to **broadcast** a "per-query scalar" to the value-column lanes
   **component by component**: `av[rr] = sub_group_broadcast(alpha_sel, alpha_lane0 + rr)` (`sdpa_ocl.cl:855-862`). → the source of the alpha[] nesting invariant (§4).
4. B is the fixed-VNNI operand (SPV_INTEL_subgroup_matrix_multiply_accumulate: fixed layout, no transpose flag, only type-reinterpretation flags).
   So V, which is read as VNNI (`transform_*` read), must go to B, and the S side becomes A. This is the transpose-symmetric counterpart of micro (A=V), and the result is the same (§2.2).

### 2.2 The sdpa_micro mapping (reference for comparison; confirmed by blob disassembly, B70 2026-07-09/10)

| GEMM | Declared | Actual DPAS | Loads | VNNI side |
|---|---|---|---|---|
| KQ (`ugemm_kq`) | A=K, B=Q, `C.layout=T` | `transC` → `problem.transpose()` **swaps** A/B → **A=Q, B=K** (M=query, N=key) | K: `load_block2d.ugm.d8` + `d32` transpose, Q: SLM | B(K) |
| VS (`ugemm_vs`) | A=V, B=S, `C.layout=N` | no swap → **A=V, B=S** (M=v_head, N=query, K=key) | V: `load_block2d.ugm.d8v` (VNNI transform), S: SLM | **A(V)** |

- Which operand is VNNI is decided by `globalCM`, not by the A/B role (`matrix_multiply.cxx:551-557`): `globalCM(VS)` → A is VNNI, `!globalCM(KQ)` → B is VNNI.
- The two DPAS instructions are identical (`dpas.8x8 hf`); only the **way the operands are fed** differs. `bdpas` (fused dequant) is **Xe3p-only**, so it is not available on Xe2/B70.
- That the two mappings are mathematically equal was confirmed with cases D/E of `test/microbench/verify_micro_kq_dpas.cpp` (maxerr=0, MEASURED B70).
- **micro's int8 V dequant "in-place on VNNI stride"** (confirmed by disassembly): `mov` (i8→w, `<4;2,1>:b` region) → stride-2 `add` (zp, even/odd separately) → `mov hf` → `rol 0x10` → stride-2 `mul` (scale).
  It dequantizes while keeping the VNNI crosspack-2 stride, so **repack 0**. OpenCL cannot control GRF regions/strides, so IGC inserts `half4 → as_int` reassembly movs → this is the **OpenCL floor** and
  cannot be imitated (→ chapter 03 dequant). sdpa_ocl instead folds the V scale into the score (P) side and removes the scale multiply from the V dequant (a different trade-off; "do not fix").

### 2.3 The sdpa_ocl_decode mapping (q=1, everywhere lane == that matrix's column)

```
KQ  S[key] = sum_d Q[d] K[key][d] :  A = Q (short, M = Q_PER_WG heads, lane = d)   B = K (int8, lane = key)   C lane = key
SV  O[d]   = sum_key P[key] V[key][d]: A = P (short, lane = key, KQ result layout as is, no shuffle)  B = V (int8, lane = d, VNNI)
```
(`sdpa_ocl_decode.cl:9-20`). Why it is the **opposite** of prefill (A=Q, B=K): there is only 1 query, so the M axis can be filled with the GQA q-heads (`Q_PER_WG`)
(`M ∈ {1,2,4,8}`, encoded as the DPAS repeat count), and a K/V tile read once is shared by several heads. The softmax uses 2 subgroup reduces instead of an SLM tile + alpha rescale.

### 2.4 The K cache layout forces the mapping
- If the PA K cache is d-major (`[head, token]`, token innermost), a coalesced read gives lane = token, which is not the axis that sdpa_ocl's A=K (lane = head dim) needs
  → K becomes a **SIMD-1 scalar gather**, which was the dominant cost of the mixed kernel (`pa-k-cache-layout-d-major`, `sdpa-ocl-int8-perf`: int8 prefill ocl 1.69x slower than micro, B580).
  Fix: relayout K into token-major pages (`OV_GPU_PA_K_TOKEN_MAJOR`; BY_CHANNEL only where `by_channel_token_major_readable()` allows it; for the current default scope see `sdpa_ocl.md` "Paged-attention cache layouts"; → chapter 02). The transpose read is 32-bit only, so an 8-bit/16-bit K cannot be read transposed.
- u4 K: a byte is a pair of adjacent channels, so lane L receives channels `(2L,2L+1)`. Permuting the contraction-axis depth is **free if A and B are permuted identically**, so
  the depth is **permuted** with `PA_K_U4_CHANNEL(db,L) = win + 2L + par` and deinterleaved only once, in the Q staging (`sdpa_ocl_qk_load.cl:15-40`; `sdpa_ocl.md` "u4 K").
  → "applying the same permutation to the contraction axis of both operands leaves the result unchanged" is a general tool for resolving layout mismatches.

---

## 3. Work distribution: subgroup/workgroup tiling

### 3.1 Terms and formulas (`sdpa_ocl_config.cl:67-77`)

| Macro | Definition | Meaning |
|---|---|---|
| `kq_sg_tile_keys` / `kq_sg_tile_queries` | injected by the host via JIT | the S tile (keys × queries) that 1 subgroup computes in KQ |
| `kq_sg_per_wg_keys` / `_queries` | 〃 | subgroup arrangement inside the WG |
| `sg_per_wg` | `pwk * pwq` | subgroups per WG = `reqd_work_group_size(SUBGROUP_SIZE, sg_per_wg, 1)` (`sdpa_ocl.cl:26-27`) |
| `kq_wg_tile_keys` (wgTK) | `tile_keys * pwk` | keys consumed by 1 iteration of the k0 loop |
| `kq_wg_tile_queries` (wgTQ) | `tile_queries * pwq` | queries owned by a WG |
| `kq_key_blocks` / `kq_query_blocks` | `tile_keys/8`, `tile_queries/SG` | size of the S_tile[mb][qb] array (float8 × count) |
| `sv_sg_tile_values` / `_scores` | value columns / queries a subgroup handles in the S*V stage | |
| `sv_sg_per_wg_values` / `_scores` | re-partitions the same subgroups for the S*V stage | |
| `DKS`, `DKS_ACTIVE` | `D_MAX/16`, `ceil(K_HEAD_SIZE/16)` (rounded up to even for u4) | number of KQ depth iterations (§9) |
| `D_MAX` | `K_HEAD_SIZE` rounded up to a power of 2 (`get_d_max`, `sdpa_gen_ocl.cpp`) | the S*V split and the alpha nesting are derived from this value → head 72 also gets `D_MAX=128` |

- Dispatch (`sdpa_gen_ocl.cpp: get_dispatch_data_func`, ~L1373): `LWS = {SG, sg_per_wg, 1}`, `GWS[0] = SG * ceil(q / wgTQ)`,
  `GWS[1] *= heads_num` (**a separate WG per head**), for plain SDPA `GWS[2] *= batch`. PA resolves the subsequences inside the kernel through `blocked_indexes_start_and_gws_mapping`.
  **PA's query-block stride must equal the JIT-ed `kq_wg_tile_queries`** (`get_query_block_size`; `make_problem` derives it once for host/JIT/dispatch). Env overrides also go through `choose_config`, so they match automatically.
- **Total subgroup count = (aligned_q / wgTQ) × heads × sg_per_wg**. This value is meaningful as a performance indicator (§6.3).

### 3.2 Kernel flow and subgroup roles (`sdpa_ocl.cl`)

```c
// sdpa_ocl.cl:118-128 -- KQ and S*V interpret the same sg_ij differently
sg_i_kq = sg_ij % kq_sg_per_wg_keys;   sg_j_kq = sg_ij / kq_sg_per_wg_keys;   // KQ: (key, query)
sg_i_sv = sg_ij / sv_sg_per_wg_values; sg_j_sv = sg_ij % sv_sg_per_wg_values;  // S*V: (score, value)
```
1. The WG cooperatively stages Q into SLM (`Q_slm`) (tiles go round-robin over the subgroups: `for (tile = sg_ij; tile < q_blocks*DKS_ACTIVE; tile += sg_per_wg)`, `sdpa_ocl.cl:~330`).
   → at head ≥ 256, where `q_blocks*DKS > sg_per_wg`, a 1:1 assignment left some tiles empty: a **silent wrong answer** (bug C7; fixed with round-robin, `sdpa-ocl-headsize-work`).
2. Upper/lower bounds of the key range (causal_k, window_k0_begin, bidir extension; `sdpa_ocl.cl:~380-430`). The absence of these bounds was the root cause of PA prefill being slower than micro (`sdpa-ocl-causal-bound`; at q=k=1024, wgTQ=32: 256 vs 144 tiles = 1.78x waste).
3. k0 loop (kq_wg_tile_keys at a time): K tile load → KQ DPAS (depth db loop) → mask + `lmax` → `__builtin_IB_atomic_max_local_f32(&S_max_slm[query], lmax)` (`:763`) →
   barrier → `exp2` / P→`S_slm` (`vstore4`) / `S_sum` → alpha rescale → S*V DPAS (cp loop, cp = 16 keys = 1 page).
4. Epilogue: sum `S_sum_slm`, then `inv_l`, store the output. A fully masked row gives 0 instead of NaN via `l > 0 ? recip(l) : 0`.
- **The key axis is a reduction axis**: `kq_sg_per_wg_keys` subgroups compute the same query set over different key ranges; the max is merged with an SLM atomic max and the sum through `S_sum_slm`
  (`sdpa_ocl.cl:763, 838`). Growing the subgroup count or the k0 tile adds merges/barriers/SLM round trips without adding useful work (§6).
- GQA: the prefill kernel uses **one WG per head** (`b0_kv = b0 / KV_GROUP_SIZE`, `sdpa_ocl.cl: b0_kv`), so WGs that use the same kv head share K/V only through L2; there is no register/SLM sharing.
  Decode puts `Q_PER_WG` q-heads into one WG to amortize the K/V loads (§8).

### 3.3 SLM size (exact formulas; the host `slm_bytes` and the kernel declarations agree; `sdpa_gen_ocl.cpp:~148`, `sdpa_ocl.cl:314-317`)

```
Q_slm  = DKS_ACTIVE * q_blocks * 8 * SG * 4B     = D_MAX*wgTQ*2  (using the DKS upper bound)
S_slm  = wgTK * wgTQ / 2 * 4B                    = wgTK*wgTQ*2
S_sum  = wgTQ * pwk * 4B ,   S_max = wgTQ * 4B
```
Default h128 (wgTQ=32, wgTK=128, pwk=8) = 8192 + 8192 + 1024 + 128 = **17536 B** (MEASURED match: diff 0 against the value cliloader reports, `sdpa-ocl-slm-vs-micro`).
The host `tiling_fits_device` checks SLM (Xe2 128KiB / xe_hpg 64KiB) and WG ≤ 1024 work-items, and `choose_config` (assert) and `supported()` (false) use the same function.

---

## 4. Tiling invariants that silently produce wrong answers

Violating them still **compiles and runs successfully**; only the accuracy comparison fails (no crash, no spill warning).

| # | Invariant | Checked at | Symptom when violated |
|---|---|---|---|
| 1 | `sv_sg_tile_values * sv_sg_per_wg_values >= V_HEAD_SIZE` (exactly: `== vd_max`) | kernel `#error` (`sdpa_ocl_config.cl:112`) + `solve_sv_split` | some value columns not computed |
| 2 | `sv_sg_tile_scores * sv_sg_per_wg_scores == kq_wg_tile_queries` | `#error` (`:109`) + `solve_sv_split` | some query rows not computed |
| 3 | `sv_sg_per_wg_values * sv_sg_per_wg_scores == sg_per_wg` (= `kq_sg_per_wg_keys * _queries`) | `#error` (`:106`) | `reqd_work_group_size` is set by the KQ side, so if the S*V product differs, **a nonexistent subgroup owns value columns** and that output is never written. An early env sweep that overrode only `kq_sg_per_wg_keys` produced **invalid timings** because of this |
| 4 | **alpha[] nesting**: in each subgroup `sg_i0_sv >= sg_j0_kq` and `sg_i0_sv + sv_sg_tile_scores - 1 < sg_j0_kq + kq_sg_tile_queries` (⟹ `sv_sg_tile_scores <= kq_sg_tile_queries`) | **no `#error`**. Only the host `solve_sv_split` checks it, in a per-subgroup loop (`sdpa_gen_ocl.cpp:~53-90`) | the alpha rescale reads `alpha[]` (which holds only the subgroup's own KQ queries) at the S*V coordinate `rel_query = sg_i0_sv + r*8 - sg_j0_kq`, goes out of range and **multiplies by the alpha of the wrong query**. MEASURED failure: (kq_tile_q=16, sv_tile_scores=32) and (32, 64) FAIL `*paged*96` accuracy; re-deriving the same KQ with `sv_tile_scores=16` PASSes |

- No. 4 must be checked **per subgroup, not just from the tile sizes** (because the two stages map `sg_ij` differently, to (key,query)/(score,value)).
  The alpha selection is implemented as a select chain to keep a runtime-indexed private array from being evicted to scratch (`sdpa_ocl.cl:855-858`; this one change removed `private memory size 128` + 4 scratch ld/st inside the k0 loop on llama-3.2-1b MIXED, about 0.8% on B70, `sdpa-ocl-mixed-kc-vc-split`).
- Other structural constraints: `kq_sg_tile_keys ∈ {16,32}` (`#error`, `:115`; k_mask/mask_tile hold at most 2 entries of 16 keys), `PAGED_ATTENTION_BLOCK_SIZE == DPAS_K` (cp block = 1 page, `:126`),
  even `DKS_ACTIVE` for u4, `S_slm` row length = `DPAS_K` (on SG8 too: 8 dwords = one block_read8 row).
- **`SDPA_OCL_KQ_*` overrides** (`SDPA_OCL_KQ_TILE_KEYS / _TILE_QUERIES / _PER_WG_KEYS / _PER_WG_QUERIES`): if any one is set, the S*V split is re-derived under invariants 1-4
  (widest `sv_sg_per_wg_values` first), with an `OPENVINO_ASSERT` if that is impossible. When unset, the tuning table is byte-identical. The tiles are JIT `#define`s, so they can be swept **without a rebuild** (confirm the choice with `SDPA_OCL_TRACE_CONFIG=1`).
  Trap: `setupvars.sh` wipes the positional arguments with `set --`, so the env was never actually applied and "different configs" gave identical results (→ chapter 05).

### 4.1 Examples of verified valid configs (D_MAX=128, wgTQ=64, REG128; `sdpa-ocl-tiling-constraints`)

| tile_q | pwk | pwq | sg/wg | wgTK | SLM(B) | sv (tv,ts,pv,ps) |
|---|---|---|---|---|---|---|
| 16 | 4 | 4 | 16 | 64 | 25856 | 32,16,4,4 |
| 64 | 4 | 1 | 4 | 64 | 25856 | 32,64,4,1 |
| 16 | 2 | 4 | 8 | 32 | 21248 | 64,16,2,4 |
| 64 | 8 | 1 | 8 | 128 | 35072 | 16,64,8,1 |
| 32 | 4 | 2 | 8 | 64 | 25856 | 32,32,4,2 |
| 32 | 8 | 2 | 16 | 128 | 35072 | 16,32,8,2 |

---

## 5. Shipped default tile table (`choose_config_kq_only`, `sdpa_gen_ocl.cpp:~180-250`)

Xe2 (SG16), `k_head_size == v_head_size`. All use `tile_keys=16`. Format: `(tile_q, pwk, pwq)` → `wgTQ/wgTK/sg_per_wg`; sv=(tv,ts,pv,ps).

| d_max | KQ (tile_q, pwk, pwq) | wgTQ / wgTK / sg | sv (tv, ts, pv, ps) | Notes |
|---|---|---|---|---|
| ≤ 32 | (32, 8, 2) | 64 / 128 / 16 | 16, 8, 2, 8 | a 32-wide head cannot be split 8 ways (tv ≥ 16) → 2x query tile |
| ≤ 64 | (32, 8, 2) | 64 / 128 / 16 | 16, 16, 4, 4 | "wide micro math" (commit `6b0e442b09`): the k tile matches micro's 128, so the accumulation order matches |
| ≤ 128 | (16, 8, 2) | 32 / 128 / 16 | 16, 16, 8, 2 | SLM 17536 B |
| ≤ 256 | (16, 8, 2) | 32 / 128 / 16 | 32, 16, 8, 2 | |
| ≤ 512 | (16, 8, 2) | 32 / 128 / 16 | 64, 16, 8, 2 | assert above 512 |
| xe_hpg | (16, 4, 2) | 32 / 64 / 8 | derived by `solve_sv_split` | DG2: 256GRF, SLM 64KiB limit |

- `k_head_size != v_head_size`: the S*V split is re-derived from `vd_max`. At `d_max ≥ 128` with query tile 32, a 32-wide V head splits only 2 ways, giving score tiles of 4/2 rows (< the DPAS minimum of 8 rows).
  **Tier 2**: retry with KQ `(16, 4, 4)` (wgTK=64, wgTQ=64) → every affected pair resolves to `(16,8,2,8)`. This hole was not harmless: the MIXED fallback `pa_multi_token` read the token-major K as d-major and produced **NaN** (`sdpa_ocl.md` "Tiling").
- **Mismatch between the current code and some memory notes**: `sdpa-ocl-mixed-kc-vc-split` says "config A (`d_max ≤ 64`: tq16/pwk4/pwq4)" ships, but the current code is the table above (pwk8); it was reverted in `6b0e442b09`. The table follows the code.
- **The winning configs of §6 below (256GRF + tq32/pwk4/pwq2, A' = tq16/pwk4/pwq4) have not been promoted to defaults since 2026-07-30 (checked in the code).** Reachable only through env. Promoting them requires re-measuring with the §10 procedure.
- **PA PREFILL on Xe2 leaves this table at `d_max` 64 / 128 / 256 / 512** (2026-10-23, `d05e5f18ae`): `solve_pa_prefill_tiling()` gives it
  `sdpa_micro`'s PA prefill workgroup, 256GRF and the pipelined loop (§6.6). The table still covers MIXED, plain SDPA, `d_max` 32 and xe_hpg.

---

## 6. Measured tiling results and causes

### 6.1 The real reason `kq_sg_tile_keys` 16→32 is slower (B70, locked at 2800MHz, head128 v128 i8-compressed prefill q=4096, `sdpa-ocl-kq-tile-keys-32-slower`)

The user's expectation: the 8b transform 32-row read (`transform_8b_32r16x1c` reads 32 keys but uses only 16) would become efficient at tile_keys=32. **The result was the opposite.**

| config | tile_keys | pwk | sg/wg | SLM | occ% | prefill min ns |
|---|---|---|---|---|---|---|
| k16pwg4 | 16 | 4 | 8 | 12928 | 54.7 | **3,526,979** (reported as faster than micro 4.17M; see the caveat below) |
| k32wg | 32 | 4 | 8 | 17024 | 58.5 | 4,346,666 |
| k16 (default) | 16 | 8 | 16 | 17536 | 63.6 | 5,102,600 |
| k32 | 32 | 8 | 16 | 25728 | 65.3 | 5,587,600 (+9.5%) |
| k16pwg16 | 16 | 16 | 32 | 26752 | 70.2 | 6,864,687 |

- Two independent axes, both "bigger is slower": axis A = `kq_wg_tile_keys` (64<128<256 at the same sg_per_wg), axis B = `sg_per_wg` (8<16<32 at the same wgTK).
  The 16→32 change grew wgTK from 128→256 (axis A) → +9.5%. `tile_keys` itself is nearly neutral (k16pwg4 3.53M < k32wg 4.35M, same sg=8).
- **The first hypothesis, "SLM → lower occupancy", was wrong (falsified)**: k32 actually has 2% fewer total body instructions (0.978x), fewer syncs, 0 spill in both (GRF 128), identical dispatch (GWS[2048×256×1], LWS[16×16×1]).
  **cliloader occupancy% is inversely correlated with speed in this kernel** (the slowest, k16pwg16, has the highest occ, 70%). EU thread residency ≠ throughput (do not trust occ% without re-measuring).
- The total WG count is the same, 2048, in every config → this is not a grid-underfill story.
- Mechanism (ASSUMED, structural explanation): the key axis is a reduction, so bigger subgroup counts/k0 tiles only add atomic max · barrier · S_slm round trips · S_tile live ranges while the useful work stays the same.
- ⚠ **Unresolved contradiction**: `sdpa-ocl-tiling-constraints` (07-30) records "the pwg4/pwg2 data in this table overrode only `kq_sg_per_wg_keys`, so they were invalid configs violating invariant 3; discard the timings",
  while `kq-tile-keys-32-slower` records "all 4 ACCURACY PASS in the head=64 ref-check". No explanation was found that satisfies both records.
  So treat **"k16pwg4 beats micro" as unconfirmed**; the general rule "smaller WGs are faster" is falsified by 6.2 below.

### 6.2 A sweep redone with valid configs: 16 subgroups win (B70, locked at 2800MHz, PA prefill `*paged*96` q=1024 32/8 heads head128 f16, with the causal_k+window bounds, min of 8 runs, 3 passes ±4%, `sdpa-ocl-tiling-constraints`)

| label | sg/wg | wgTQ | wgTK | SLM | min ns | vs micro |
|---|---|---|---|---|---|---|
| default | 16 | 32 | 128 | 17536 | 3,273,229 | 1.087x |
| **A' (tq16, pwk4, pwq4)** | **16** | **64** | **64** | 25856 | **2,268,020** | **0.753x** |
| E (pwk4, pwq2) | 8 | 32 | 64 | 12928 | 2,795,208 | 0.928x |
| C (tq64, pwk4, pwq1) | 4 | 64 | 64 | 25856 | 3,892,187 | 1.292x |
| D (tq64, pwk8, pwq1) | 8 | 64 | 128 | 35072 | 4,189,687 | 1.391x |
| micro | 32 | 128 | 256 | 107008 | 3,011,041 | — |

- In this shape **sg/wg=16 (A') beats 8 (D,E) and 4 (C)** → 6.1's "smaller WG is faster" is **shape-dependent, not a general rule**.
  Winning condition: **wgTQ=64 and wgTK=64 together, in a full 16-subgroup WG** (causal efficiency 50%→66% + a small k0 tile). Keeping wgTQ=64 but going back to wgTK=128 (D) is 1.85x slower, and reducing sg at wgTQ=64 (C) is 1.72x slower.
- The full 143-test PA suite PASSes with A' (all heads 32-512, prefill/generate/mixed) → it was a default candidate but **was not promoted**.

### 6.3 The sweet spot of the total subgroup count (gpt-oss-20b u4, B70 total execution ns, `sdpa-ocl-u4-head64-page-read`)

| config | total subgroups | total execution ns |
|---|---|---|
| 1D page read + default (tq16) | 8192 | 632,343,047 |
| (e) `KQ_TILE_QUERIES=32` | 4096 | **397,934,594** (faster than micro 401,436,373) |
| (h) (e)+`PER_WG_KEYS=4` (sg 16→8) | 2048 | 427,729,494 |

(h) lost even though it has the fewest loads per unit of work (80 vs 288 block reads / 4096 key-query pairs) → **static "messages per work" alone is a wrong metric**. Halving the thread count weighed more.
This (e) is reflected in the current `d_max ≤ 64` default as `tile_queries=32`.

### 6.4 Why a narrow key tile wins in MIXED (PA with past) (llama-3.2-1b, past=32 q=1027, u4, B70; `sdpa-ocl-mixed-kc-vc-split`)

| config | sg/wg | wgK | SLM | SPILL | ns/call (at the time of the page bound) |
|---|---|---|---|---|---|
| default | 16 | 128 | 26880 | – | 153,227 |
| **A** | 16 | 64 | 17664 | – | **146,417** |
| B | 8 | 64 | 17664 | 512 | 157,877 |
| C | 8 | 128 | 26880 | 11776 | 372,628 |

A vs C is the key: you win by **narrowing the key tile while keeping the threads**. Cause: in the one k0 tile on the cache/current boundary, the cache-side block is ~2.2x slower than the current (Kc) side, and the KQ stage ends with a WG barrier, so
every subgroup pays for the slowest one. Halving wgK halves the idle time (ASSUMED mechanism + consistent with the measurement). On the V side the cp loop is subgroup-local and has no barrier, so the split gain is kept in full.
For B/C **ocloc predicted spill 0, yet they spill at runtime** (§7.3).

### 6.5 256GRF + tile pairs (llama-3.1-8b f16, q=4096 prefill, B70 2800MHz, `_prefill` device time, 3 passes <0.5%, `sdpa-ocl-beats-micro-256grf`)

| config | sg | wgTQ | wgTK | SLM | spill | avg ns |
|---|---|---|---|---|---|---|
| sdpa_micro | – | – | – | – | – | 1,930,454 |
| ocl default (REG128) | 16 | 32 | 128 | 17536 | 0 | 2,181,376 (13.0% slower) |
| **256GRF + tq32/pwk4/pwq2** | 8 | 64 | 64 | 25856 | 0 | **1,794,065** (7.0% faster; e2e 1st token 466.98→462.85 ms) |
| 256GRF tq32 pwk4 pwq4 | 16 | 128 | 64 | 51712 | 0 | 1,874,063 |
| 256GRF tq64 pwk8 pwq1 | 8 | 64 | 128 | 35072 | 0 | 1,890,184 |
| 256GRF default tiles | 16 | 32 | 128 | 17536 | 0 | 2,725,145 (**25% slower**) |
| 256GRF T128k256 | 16 | 128 | 256 | 107008 | 3328 | 4,148,942 |
| 256GRF T128k128 | 8 | 128 | 128 | 70144 | 13952 | 13,795,231 |

- The full 143-test PA suite PASSes. SDPA prefill is ~10-15% of the 1st-token latency (69.8/475 ms) → a 7% kernel gain ≈ 0.9% e2e. **Per-kernel time is the metric**; e2e barely moves.
- **The k0 iteration count is not the goal**: T128k256 has the fewest iterations (272 = same as micro) but is 2.2x slower because of spill.
- Root cause (ISA region profile, default-tile k0 loop body 379 inst): KQ 26.1%, **softmax 47.5%** (causal mask+lmax 61, barrier+S_max 13, exp+lsum 49, S→SLM VNNI store 57), SV prep 12.4%, SV dpas 14%.
  dpas is only 32 inst (8.4%). ocl does 2112 k0 iterations per head vs micro 272 (7.76x), while it actually computes 2.9% fewer score elements → the problem is the **fixed per-iteration overhead** (atomic fmax → barrier → S_max read, S→SLM, alpha). (→ chapter 04)

---

### 6.6 PA PREFILL: `sdpa_micro`'s workgroup per head size (2026-10-23, B70 pinned 2800 MHz, cliloader `_prefill` device time, `sdpa-ocl-longctx-prefill`)

Symptom: the first token of every model fell further behind master as the prompt grew (qwen3-8b e2e +2.0% at 2K -> +24.0% at 32K).
The kernel ratio was **constant** (1.30x at 2K, 1.39-1.42x from 4K to 32K); only attention's share of the first token grew (13% at 2K ->
66% at 32K, quadratic vs linear), so a 1K-prompt benchmark cannot show it (-> chapter 05, prompt-length sweep).
Cause: the table tile (128 keys x 32 queries, REG128) reads every K/V row once per 32 queries — 4x as often as micro's 256 x 128 at
head 128 — and does a quarter of the DPAS per barrier.

| d_max | KQ (tk, tq, pwk, pwq) | WG keys x queries / SG | SLM B | `sdpa_micro` | kernel vs micro, pinned (model, 2K..32K) |
|---|---|---|---|---|---|
| 64 | 16, 64, 8, 2 | 128 x 128 / 16 | 53,760 | `xehpc_h64`, same WG | 0.85-0.88 (llama-3.2-1b), 0.76-0.89 (gpt-oss sink+window) |
| 128 | 16, 64, 16, 2 | 256 x 128 / 32 | 107,008 | `xehpc_h128`, same | 1.008 -> 0.913 (qwen3-8b) |
| 256 | 16, 32, 8, 4 | 128 x 128 / 32 | 102,912 | `xehpc_h256`, same | 0.774 (gemma-2-9b), 0.81-0.83 (qwen3.5-9b) |
| 512 | 16, 16, 8, 4 | 128 x 64 / 32 | 84,224 | `xehpc_h512` = (32, 16, 8, 4), 256 x 64 | 1.006 / 0.956 / 0.931 / 0.962 / 1.012 (gemma-4-e2b) |

- **Mirror the reference's workgroup tile, not only its subgroup tile**: K/V is re-read `seq / wgTQ` times and the per-barrier work is `wgTK x wgTQ`.
- The tile alone was 1.03-1.14x micro at head 128; four loop changes took it to 0.91-0.93x (branchless exp, V prefetch, split S_max barrier,
  last-tile mask; chapters 02 §7.5/§8, 03 §1.4/§2.7). Tile and loop are one tuning unit.
- `MICRO_MATH` reproduces micro bit for bit only with micro's **key** tile (the online-softmax rescale points): d_max 64/128/256 give MD5 =
  master 20/20 on every model without window/sink/token types. d_max 512's 128-key tile is not micro's 256 -> not micro-identical, but its
  128-key WG and 64-value S*V tile equal the old table's -> MD5 equal to the table tiling 20/20 (positive control `SDPA_OCL_MICRO_MATH=0` differs).
- **256GRF halves the dispatchable work-group**: 16 x 4 subgroups (1024 work-items) at 256GRF is `CL_INVALID_WORK_GROUP_SIZE` (-54, MEASURED),
  although `tiling_fits_device()` allows 1024 (§14). Keep 256GRF tilings at <= 32 SG16 subgroups.

d_max 512 sweep (gemma-4-e2b 8/1 heads, 7 global layers; per call, micro 680 us at 2K / 9,147-9,195 us at 8K; `test/longctx/g4_an.py`):

| KQ tile | GRF | loop changes | 2K us | 8K us | Note |
|---|---|---|---|---|---|
| table (16, 16, 8, 2) | 128 | - | 2,341 | 34,411 | SPILL 11,648 (3.8x micro): the pre-2026-10-23 default |
| table | 256 | - | 721 | 9,414 | SPILL 1,344 |
| table | 256 | all four | 766 | 10,324 | V prefetch hurts at 512 (02 §7.5) |
| **16, 16, 8, 4** | 256 | all but V prefetch | **680** | **8,585** | shipped (32K 1.012x micro) |
| 16, 16, 8, 4 | 256 | all four | 696 | 8,869 | |
| 16, 16, 16, 2 | 256 | all four | 759 | 9,731 | 256 x 32 |
| 16, 16, 16, 4 | 256 | - | - | - | 1024 work-items -> -54 |
| 16, 16, 4, 4 / 16, 32, 8, 2 | 256 | all four | 5,186 / 3,413 | 68,995 / 43,147 | SPILL 17,536 / 17,024 |
| 32, 16, 8, 4 (micro's) | 256 | all but V prefetch | 607 | 7,232 | 0.79x micro (32K 135.6 vs 162.4 ms); **not shipped**: tile_keys 32 (§13) |

## 7. GRF mode and spill

### 7.1 How it is selected
- `SDPA_OCL_256GRF=1` → `-cl-intel-256-GRF-per-thread` (`SDPAOclGenerator::get_build_options`, `sdpa_gen_ocl.cpp:1041-1042`). **Always on for xe_hpg** (at 128GRF every measured tile spills on DG2).
  Decode uses `SDPA_OCL_DECODE_256GRF`. The Xe2 default **stays at 128GRF** (with the default tiles 256 is actually a loss, see below).
- sdpa_micro always runs at REG256 when the microkernel asks for more than 128. So comparing ocl REG128 vs micro REG256 was unfair from the start, and **256GRF unblocked the large tiles that spill had been blocking**
  (every config with `sv_sg_tile_scores >= 64` spills 3.8-19KB at 128GRF, but has spill 0 at 256GRF).
- Since 2026-10-23 PA PREFILL on Xe2 at `d_max` 64-512 compiles with 256GRF by default (`tiling.large_grf` from `solve_pa_prefill_tiling()`,
  §6.6); `SDPA_OCL_256GRF` is now an int knob and `0` turns it off there.
- GRF size: Xe2 64B, Xe-HPG 32B → DG2 256GRF = 8KB/thread = the same bytes as Xe2 128GRF. The thread count per EU is halved.

### 7.2 256GRF is not free (MEASURED)
| Shape | 128GRF | 256GRF | Source |
|---|---|---|---|
| llama-3.1-8b f16 prefill default tiles | 2,181,376 | 2,725,145 (+25%) | B70, §6.5 |
| gemma-4 head 72 prefill (scalar-gather state) | 9,570,971 (spill 2688B) | 14,119,903 (spill 0, **+47% slower**) | B70, `test/sdpa_ocl_head72_analysis.md` §2.7 |
| decode gemma-4 head512 sg8 | – | 157.1M ≈ sg16's 158.6M | B70, `sdpa-ocl-decode-tiling-sg-per-wg` |
| decode gemma-4 head256 | 408.9M | 505.3M (loss) | 〃 |

→ 256GRF pays off only when the gain from removing spill exceeds **the cost of halving the thread count**. It must be tuned **as a pair** with large (spill-inducing) tiles. At head 72 the spill was a **symptom**, not the cause
(128 scalar gathers → the LSC message count was the cost; once the loads became block reads, the spill 2688→0 disappeared as a side effect).

### 7.3 Limits of spill prediction (MEASURED)
- The spill prediction of `ocloc -device bmg` is **useless here**: all 8 256GRF candidates had ocloc spill=0, yet 4 spilled 3.3-19KB at runtime (`sdpa-ocl-beats-micro-256grf`). Same for MIXED configs B/C.
  **Trust only the runtime cliloader `SPILL=`** (→ chapter 04). The instruction/dpas/send counts of the ocloc ISA, on the other hand, match the runtime exactly (k16: dpas 32=32, send.ugm 98=98, inst 3039=3039, same hash).
- In decode the spill gate is implemented entirely as a host model (`live_grf_estimate`, budget 112 GRF) (§8.2).

---

## 8. Tiling for decode and MIXED

### 8.1 decode: `SG_PER_WG` matters more than M (gemma-4-26b-a4b-it, B70, sum of attention device time in ns, `sdpa-ocl-decode-tiling-sg-per-wg`)

| | head512 | head256 | sum | e2e |
|---|---|---|---|---|
| pa_opt (baseline) | 197.9M | 520.2M | 718.1M | 11.43 ms |
| old default (sg8, M=8) | 623.2M | – | 1342M | 12.14 ms |
| **new auto** | **118.3M** | **407.6M** | **525.9M (0.73x)** | 11.40 ms |

- A subgroup = 1 thread. sg8 has 1/4 the threads of pa_opt (`LWS[1x1x512]` = 32 SIMD16 threads). At M=1, relative to sg8, **sg16 = 2.07x (head512), 1.26x (head256) faster, sg4 = 2.2x slower** (monotonic, steep in both directions).
- **How to diagnose TLP-bound**: going from 2p→5p partitions (2.5x the work), the device time stays almost unchanged at 121,728→122,290 ns (+0.5%) while the baseline goes +103% → "absorbing 2.5x the work for free" = a sign of idle thread slots.
- Rule `get_sg_per_wg`: `(v_head/16 >= 16) ? 16 : 8` (`sdpa_gen_ocl_decode.cpp:142-147`). Reason: in S*V each subgroup owns a V head-dim tile; when `SG_PER_WG > V_TILES`, the extra subgroups **split the key axis**
  (`SV_KEY_SGS>1`), which adds `slm_out` staging + 1 barrier. That sg16 was recorded as "neutral or a loss" at head128 (V_TILES 8) and as 2.07x at head 256/512 are both true, and `V_TILES` is the discriminator. (Superseded on 2026-10-22 by the rule in §8.3.)
- **M (=Q_PER_WG) is limited by the register budget**: the `q_reg[M][K_TILES]` halves alone fill all 128 GRF at head512 M=8 → **SPILL=34432, 3.14x slower than pa_opt**.
  `live_grf_estimate()` budget 112 (`sdpa_gen_ocl_decode.cpp:~70-100`). Calibration table (u4 BY_CHANNEL): h512 s8 M=8/4/2/1 = estimated 332/220/164/136 → measured spill 34432/15936/6400/0;
  h512 s16 M=2/1 = 126/100 → 640/0; h256 s16 M=2 = 94 → 0. **Differences below about 15 GRF cannot be resolved** (136 has spill 0, 126 has spill 640) → it is only a coarse gate. The model is conservative (h256 sg8 M=2 = 116 is rejected even though it has spill 0).
  (2026-10-28: accepting M = 1 here was the cause of gemma-4's long-context decode gap; the cap now triggers a low-register layout instead, §8.4.)
- `SDPA_OCL_DECODE_M` bypasses the register cap but not the SLM clamp (`SDPA_OCL_DECODE_M=8` is silently ignored if `kv_group_size < 8` → it may not "reproduce the old default").
- **Why the diagnosis needed no rebuild**: M / SG_PER_WG / 256GRF / K_2D are exposed as env. 13 configs were diagnosed with 0 rebuilds.
- Trap: without `-lc '{"ATTENTION_BACKEND" : "PA"}'` in llm_bench, gemma-4 takes the stateful SDPA path, `sdpa_ocl_decode` is never dispatched, and every env config measures byte-identical (5 sweeps wasted).
  "Before interpreting an A/B, confirm that the code path is live" (→ chapter 05).
- General lesson: **occupancy has two axes (WG count vs threads per WG) that cannot substitute for each other**; "a knob that was neutral in one shape is not a dead knob" (SG_PER_WG=16 neutral at head128, 2.07x at head512).
- Remaining uncertainty: the 1.37x device-time gain becomes parity in e2e (pa_opt gains ~0.18 ms/token more, cause not explained; 29% of the wall clock is not device time, 823 enqueues/token). It may not be a problem kernel tiling can solve.

### 8.2 Other decode facts
- The probabilities P are stored in SLM as **key index × head vector component** (storing them head-major means 64 reads of 32B → SLM loads 44→72, instCount +12%).
- The partition page table is read once, 1 chunk per lane, and broadcast (looking it up at every use costs 18 scalar loads).
- V prefetch distance: 2D block prefetch for V only (+2.1% llama-3.1-8b head128 M=4, 1.1194e9 → 1.0958e9 ns at dist 4); K prefetch is a 3.6% loss (KQ already has KEY_GROUPS independent DPAS chains) (→ chapter 02).
- "2x traffic → +23%, cutting instructions 6.8% · SLM traffic 86% → +1.2%": memory-level parallelism was the limit, and every occupancy knob (SG_PER_WG 2/4/16, 256GRF) was neutral or worse (llama head128).

### 8.3 decode: SG and M are one pair; the discriminators are the cache type and the threads per partition (B70, 2026-10-22, sum of GENERATE stage-0 device time, `sdpa-ocl-decode-sg-m-rule`)

- **First check the real KV-cache precision.** A PagedAttention model with 4-bit weights gets a **u4** cache by default (K BY_CHANNEL, V BY_TOKEN; `execution_config.cpp` ~line 304); i8 BY_CHANNEL is the default only for models without 4-bit weights. All 2026-10-22 runs of lfm2.5, llama-3.1-8b, qwen3-8b, phi-4-mini, llama-3.2-1b and minicpm4 (all 4-bit) were u4; a first write-up called them i8 and had to be corrected. Force a cache with `OV_KV_CACHE_PRECISION=i8` and confirm it by the changed kernel hash in the cliloader report.
- **Changing SG changes M (confound).** `live_grf_estimate()` has `key_groups = 256 / SG / 16`, so 16 subgroups admit a larger M under the budget of 112. u4 reads four K tiles per message (`k_tiles_per_read = 4`), so head 128 with kv group 4 (llama-3.1-8b, qwen3-8b, 32/8) scores 124 at SG8/M4 -> **M = 2**, and 106 at SG16/M4 -> M = 4. The first "SG16 = 0.8x" reading was SG and M changing together. Recover the real (SG, M) from the cliloader `-dv` line: `LWS[16 x SG x 1]`, `GWS[16 x SG*head_groups x P]`, `SLM=` (M*512 + slm_out + 8*SG*M bytes) (chapter 07 §7.1). u4 SG8/M4 did not spill and was the fastest config at 4K, so the estimate is conservative there.
- **Wave model.** B70 runs about 2048 threads at once (32 Xe-cores x 64 threads at 128 GRF). Threads per partition = kv_heads x ceil(G/M) x SG. At equal threads per partition the wave count is the same, so the config with the shorter per-wave time wins. Per-wave times derived from the measurements (u4, head 128, llama): S8M4 ~25 us, S8M2 ~17.5, S16M4 ~13.5, S16M2 ~11.5. Example: llama-3.1-8b 4K at S8M2 goes from P16 to P17 (256 -> 272 workgroups) and the per-call time from 23.9 to 35.4 us (+48%): the second wave.
- **u4 table** (total ms, `-ic 128`):

| | S8M2 (old default) | S8M4 | S16M2 | S16M4 |
|---|---|---|---|---|
| llama-3.1-8b 32/1024 prompts (P1-5) | 219.8 | 304.2 | 180.9 | 180.0 |
| llama-3.1-8b 4K (P17) | 285.5 | 208.1 | 287.9 | 225.3 |
| qwen3-8b 8K (P33) | 505.8 | 448.6 | 519.9 | 386.7 |

  phi-4-mini (24/8, kv group 3, M <= 2 before the change): 32/1024 prompts S8M1 217.7, S8M2 221.7, S16M1 194.8, S16M2 182.9; 4K 298.7 / 195.1 / 295.8 / 229.5. S16M2 at 4K is 1.18x slower than S8M2: 16 subgroups double the threads per partition when M cannot grow.
- **i8 behaves differently.** S16/S8 at equal M: llama (M4) 0.85x / 1.29x / 1.01x at 32/1024 prompts / 4K / 8K; phi-4-mini (M2) 0.85x / 1.17x; lfm2.5 head 64 (M4) 0.89x / 1.26x. A u4 workgroup is slow (dequant, and scalar K/V loads at head 64, whose 32 B row misses the block2d rule), so halving the keys per subgroup pays even when the GPU is full; an i8 workgroup is fast, and once the GPU is full 16 subgroups only add waves. Head 96 (phi-3.5, u4, kv group 1): 0.97x / 0.96x, and worse at some partition counts.
- **A larger M** shares each K/V page across more heads, so the total work (loads + dequant) falls, but at short context it halves the workgroups and loses parallelism (u4 S8M4 at 32/1024 prompts: 304.2 vs S8M2 219.8).
- **Shipped rule**: 16 subgroups iff V_TILES >= 16, or (u4 && head 64 or 128 && M >= 4 at 16 subgroups); otherwise 8. M rounds the kv group **up** to a power of two (G3 -> M4); the surplus row takes the existing leftover path (Q load clamped to `head_base`, sink clamped, every store skips `m >= heads_this_wg`), and the kernel `#error` is now `Q_PER_WG < 2 * KV_HEADS_GROUP_SIZE`.
- **Result** (new / old GENERATE device time): u4 llama 0.821 / 0.792 (32/1024 prompts / 4K), qwen3 8K 0.767, phi-4-mini 0.818 / 0.823; i8 phi-4-mini 1.003 / 0.904; i8 lfm2.5 1.105 / 0.801 (back to the original 8 subgroups); unchanged configs 1.00. The M round-up alone (u4 phi-4-mini S16M2 -> S16M4) at 4K: 228.4 -> 160.0 (0.70x). End-to-end 2nd token: llama -0.9/-0.8% (4K -2.1%), qwen3 -0.9/-1.6%, phi-4-mini -1.7/-1.3% (4K -1.9%); 8B-class decode is dominated by the FC weights.
- **Correctness oracle**: heads are independent DPAS rows, so a change of M alone must be bit-identical (i8 phi-4-mini M2 -> M4 and u4 phi-4-mini S16M2 -> S16M4: identical generated-text MD5); a change of SG reorders the reductions and may change the MD5 (chapter 03 §9.3, bit-preserving vs order-changing). The standalone probe gained 12 M > G cases (G3/M4 for f16, i8, i8 BY_CHANNEL and u4; G5/6/7 with M8; up to 17 partitions; sliding window; sinks), all passing.
- **Open**: the best SG depends on the partition count, but a compiled kernel has one value; two compiled variants chosen per dispatch from `num_of_partitions` would keep the i8 short-context gain (0.85-0.89x) without the 4K loss.

### 8.4 decode: when the register cap holds M below the kv group, move data out of registers (B70 at 1800 MHz, 2026-10-28, `sdpa-ocl-decode-low-reg-layout`)

- **Symptom**: gemma-4-26b-a4b-it 2nd token against master rose with the prompt, 0.99x at 278 tokens to 1.13x at 133K (llm_bench, daily prompts). A ratio that grows with the context is a cost that grows with the keys.
- **Attribution** (chapter 07 §7.1, per-decode-token subtraction of two cliloader runs): at 32K only the 5 global layers (head 512, 2 kv heads, kv group 8, u4) differed, `sdpa_ocl_decode` 712.8 us/call against `pa_gqa_single_token` 441.9. `GWS[16 x 256 x 131] LWS[16 x 16]` = 16 head groups = **M = 1**: every page was read and u4-dequantized 8 times per kv head (`pa_gqa`: twice, `HEADS_PER_WI` 4). The 25 SWA layers were already faster (22.7 vs 29.4 us). At short context M = 1 wins by parallelism; once the GPU is full the redundant work dominates, hence the growth.
- **Env sweep first** (no rebuild): at 128 GRF M2/M4/M8 spilled 640/7552/17408 B; at 256 GRF M8 took 269 us/call even with an 8.6 KB spill. Amortization wins, so the register file is the limit, not the idea.
- **What filled the registers was invariant data**: `q_reg[M][K_TILES]` (128 GRF at M = 8) and the BY_CHANNEL comp `k_sc/k_zp[KEY_GROUPS][K_TILES]`. Four jit paths move data only, so the output is **bit-identical** at any M (each head is its own DPAS row; every product, fma and reduce keeps its order):

| path | what moves | probe, head 512 M = 8, 33.4K keys |
|---|---|---|
| (plain layout) | – | 630 us, spill 17.7 KB (17,408 B in the model) |
| `K_COMP_PER_READ` | BY_CHANNEL (scale, zp) of one K read loaded with it; `k_corr` accumulated in the KQ loop, same t order | 394 us |
| `+ Q_IN_SLM` | scaled Q in SLM as one `Q_PER_WG`-wide vector per (tile, lane), read back per A operand | 191 us, spill 0 |
| `+ V_TWIN_PAIR` | u4 V tiles cd / cd + V_READS share one read (legal only with V_TILES >= 2 SG, so no reduction) | 177 us |
| `+ V_COMP_IN_SLM` | V (scale, zp) loaded once per page by the KQ subgroup that owns it; scale folded into `slm_p` (same f16 multiply), zp staged | **161 us** |

  `V_COMP_IN_SLM` is the large one at short context (M = 8, 300 keys: 38 -> 24 us): every S*V subgroup used to load every page's pair inside the chunk loop and wait for it, and the V prefetch covers only the data rows. Instruction counts do not show such a serialized load.
- **Rule** (`q_tiling_for()`): when `live_grf_estimate()` caps M below the rounded-up kv group, take M from `low_reg_grf_estimate()` (no `K_TILES` term; budget 160 between 148 spill-free and 184 = 896 B spill) with the four paths where each applies; not at head 64 once the plain M is 4 (a page is read at most twice there; i8 gpt-oss / minicpm4-0.5b lost 6-15% up to 2K). Unchanged configs compile to the identical ISA (only IGC's source-hash `mov null` lines differ).
- **Result**: probe head 512 per call 28/36/74/122/205/362/679 -> 24/25/27/34/48/94/161 us (300 ... 33.4K keys, L2 flushed). Real 32K: 161.5 us/call, decode device time 13.02 -> 10.20 ms/token (master 11.81). End to end new/master: 0.986, 0.961, 0.952, 0.963, 0.945, 0.931, 0.888, 0.843 (278, 1K, 2K, 4K, 8K, 16K, 32K, 128K). The other shapes that switch (gemma-4-e2b global + SWA, minicpm4-8b, qwen3.5-9b, qwen3.6-35b-a3b, qwen3.8-27b): kernel 0.46-0.97x, end to end 0.99-1.00x short and 0.80-0.96x long.
- **Lessons**: (1) a host cap on the amortization factor trades a fixed register problem for traffic that grows with the context; look at what fills the registers before accepting it. (2) Keep data-movement changes bit-identical so the bitwise oracle (chapter 05) carries the whole correctness argument. (3) The standalone probe with an L2 flush reproduced the in-model per-call times (M1 679 vs 714 us, M8 630 vs 624), so all of the above ran without a plugin rebuild; at short context the real KV is cache-resident, so confirm short prompts on the model.

---

## 9. Head-size generality (32 ~ 512, 72, 80, 48/96)

- Range: `d_max ≤ 512`, assert above that (`choose_config_kq_only`). Head 32 fails some tests, but **sdpa_micro fails the same cases identically** (a test/reference problem unrelated to the kernel, B580 record).
- **Q staging bug at head 256/512 (C7)**: the 1:1 `sg_ij` assignment left tiles empty when `q_blocks*DKS > sg_per_wg` → fixed with round-robin. After the fix h64/128/256/512 PASS (B580, `sdpa-ocl-headsize-work`).
- **`D_MAX` padding cost**: head 72 → `D_MAX=128`; 3 of the 8 depth tiles are entirely 0, yet their DPAS still runs. `DKS_ACTIVE = ceil(K_HEAD_SIZE/16)` shrinks **only the KQ depth loop and the Q staging** and keeps `D_MAX` (because the S*V split and the alpha nesting are derived from that value).
  micro does not pay this cost because `ugemm_kq` takes the contraction length as a runtime argument. For u4 the depth permutation is grouped in pairs, so `DKS_ACTIVE` is rounded up to even.
- **The head 72 case (gemma-4 SigLIP vision prefill, 16 heads q=k≈2528, B70, `test/sdpa_ocl_head72_analysis.md`, commit `191722ba0e`)**:
  8.11x slower (9,570,971 vs micro 1,179,897 ns) → **1.14x faster** than micro (1,030,869 ns). Breakdown: (1) the block2d gate lumped three hardware rules (width ≥64B, pitch %16, base 64B-aligned) into `row_bytes % 64`,
  so it was entirely off at head 72 (144B rows) → K/V became 256 per-lane scalar gathers → relaxed to `%16` + a base fixup inside the kernel (`prem = base & 63`, x correction, width extension) → **6.96x** (forcing KV_2D alone: 1.375M ns).
  (2) `DKS_ACTIVE` → another 1.33x (more than the -19% of work: Q_slm 8192→5120, WG SLM 17536→14464 B, resident WGs 7→9). Q/A 2D is prologue/epilogue, so +0.35% (noise).
  Lesson: tiles/SLM/occupancy/WG layout were byte-identical between the two kernels, and **the cause was a one-line over-strict host gate** — "before suspecting the tiling, check that the IO path is on".
  Paradox: `DKS_ACTIVE` actually **increases the spill of the scalar fallback path** (DKS_ACTIVE 8/6/5 → spill 2688/13568/16064, instCount 7043→5631; ocloc static, not measured on device; applies only to head 48/96).
- Heads that are not a multiple of 64 (80/96/112 etc.) are legal under the block2d spec, but our helper is stricter (§ `sdpa-ocl-8b-transform-32row-min`; → chapter 02). The `CL_OUT_OF_RESOURCES` in the head 486/387 `*ScaledAttn*` sweep is suite-level resource pressure at SLM ~41KB (does not reproduce alone, unrelated to this change): but since it does not reproduce alone, **do not attribute it without a two-sided (on/off) control**.
- decode: the larger the head, the more `V_TILES`, so sg16 becomes free (§8.1).
- head 64 u4: the 32B row is below the block2d minimum (64B), so it is solved with a 1D whole-page read (`uc16`) → gpt-oss 3.92x (`sdpa-ocl-u4-head64-page-read`, → chapter 02).

---

## 10. SLM vs SLM-free design (compared with micro)

PA prefill head128 (`*paged*96`), B70, **computed exactly from the source formulas, diff 0 against the cliloader values** (`sdpa-ocl-slm-vs-micro`):

| Buffer | micro | ocl |
|---|---|---|
| S_slm | 256·128·2 = 65,536 | 128·32/2·4 = 8,192 |
| Q_slm | 128·128·2 = 32,768 | 8,192 |
| S_sum | 8,192 | 1,024 |
| S_max | 512 | 128 |
| ugemm SLM | **0** | n/a |
| Total | **107,008** | **17,536** (6.1x smaller) |

- The difference is **exactly two tiles** (micro: query 128 / k0 256, ocl: 32 / 128). micro's GEMM itself **uses no SLM at all** — it streams K/V from global straight into the systolic array (prefetch only). The 107KB is 100% the wrapper's Q/S staging.
  "micro uses a lot of SLM" is a fact about the **tile size**, not about the GEMM implementation.
- occupancy (Xe2 Xe-core SLM 128KB): micro fits 1 WG/Xe-core (82% SLM), ocl 7 (ASSUMED arithmetic; per §6.1, occ% is not a performance metric). Candidate reasons micro still won: causal efficiency (80% vs 50%), GRF 256 + spliced ukernel ILP.
- **Why ocl uses SLM**: (1) Q: the whole WG does the B-operand VNNI packing only once (prologue), (2) S: the (lane = query) → (lane = key) transpose (§2.1), (3) S_max/S_sum: merging across the key-split subgroups.
  Reading S*V's A from SLM with `block_read8` is a cheap way to transpose the layout, not GEMM operand feeding.
- To reduce SLM (ASSUMED, not tried): have one subgroup take all keys without cross-subgroup merging (pwk=1), or consume S in registers the micro way. Note, though, that in §6 reducing pwk was already a gain.

---

## 11. xe_hpg (DG2, SG8) rewrite details

Current state: xe_hpg takes the sdpa_ocl lane only with `TEST_USE_SDPA_OCL_HPG=1`. `kHpgTiersReady = PLAIN_F16_STATIC | PLAIN_EXT | PLAIN_I8` (`sdpa_ocl_hpg.hpp:33`), and everything else is routed to micro
(`sdpa_ocl_hpg.hpp`, `sdpa_ocl.md` "xe_hpg bring-up", recent commits `10e4f00b68`, `087d2bd171`, `b739dc5881`). **sdpa_ocl vs micro performance numbers on DG2 are not in this repository/PC** (the S6/S7 measurements are on the DG2 PC — unverified). What follows centres on the S0/S2 real-hardware results.

### 11.1 Hardware facts (S0/S2, MEASURED DG2 A770 IP12.55.8, driver 25.13.33276.16)
| Item | Result |
|---|---|
| SG8 DPAS f16/bf16 × M=1/2/4/8 | 16/16 compile successfully (`dpas.8xM`, exec8) |
| SG16 `short8` A | compiles + **no DPAS** (silently dropped) |
| 2D block IO / prefetch / `cl_intel_subgroup_buffer_prefetch` | **unsupported (compile rejected)**. micro also uses block2d only on xe_hpc and later. → the 2D paths must be turned off with a JIT switch (`#ifdef` cannot turn them off: the pragma only warns, the macro stays undefined; `block2d_io_allowed()`) |
| global uint/ushort block IO | works. But **`block_read uint` is wrong at an odd row pitch** (err 49), and `block_read_us` at a 2B offset is wrong → block read only when the pitch is even + 4B-aligned, otherwise fall back |
| local block IO | works (`addr = base + j*8 + lane`) |
| local fmax atomic, split barrier, sub_group operations | PASS |
| `-cl-intel-256-GRF-per-thread` | numGRF 128→256 |
| SLM / max WG | 64KiB / 1024 |
| generic `prefetch()` | no effect |

### 11.2 Only 3 places had to be rewritten for SG8 (the rest is shared with SG16 thanks to the lane = query / value structure)
1. **KQ's K A-operand load**: `ushort8 k_raw` (lane = head dim) → `int8`, component = key, lane = dword (2 head dims). `k_tile_dword` (`sdpa_ocl_qk_load.cl:487`):
   Per key row, 1 `intel_sub_group_block_read((const __global uint *)row)` (8 lanes × 4B = 32B). Guards: `dword_ok` (even pitch + 4B-aligned base, proven 1x per WG), `whole_tile` (head_base+16 ≤ d). Tails/misalignment use a fallback that reads each lane's two halves as an aligned dword (two `ushort` loads are **miscompiled** on DG2 for some pitches/views (query block 1 wrong) → hence the dword).
   For the head tail it is enough to guard **only one of** K or Q with 0 (S2). NaN defence: halves past `d` and keys past `k` stay 0 — even if Q is 0, `0 * NaN = NaN`, so the 0 on the K side is required.
2. **S*V's pA read**: `block_read_us8` → `as_int8(intel_sub_group_block_read8((local uint *)&S_slm[...]))` (`sdpa_ocl.cl:916`). One S_slm row = DPAS_K halves = 8 dwords = one SG8 block read row.
3. **The A type of the DPAS call** (`DPAS_A_T`).
The rest (Q staging `vload16` fallback, V `v_tile_gather`, alpha `sub_group_broadcast`, O scalar store, sink/bidir/qq_bias/causal) is unchanged. **The softmax state is lane = query, so it is the same on SG8** — this design choice bought the portability.
- `plain i8`: `k_tile_dword_i8` dequantizes `(q - zp) * scale` per element directly into the dword A (byte loads, no alignment assumption); V uses the common scalar `v_tile_gather`.
- Negative controls: `SDPA_OCL_NEG_SG8=1..6` (swap the K pair order / transposed pA read / swap S_slm pairs / force the misaligned K fallback (=4, must PASS) / wrong mask lane width) — each is meant to **make the sharp-softmax test fail**. If it passes, the test cannot observe that feature (→ chapter 05).

### 11.3 SG8 tiles/GRF (S2 microbench, MEASURED DG2; `test/sdpa_ocl_xe_hpg/s2/S2_RESULTS.md` — D=128 mini SDPA, times are t_med/min µs)
- 256GRF, a 16×16 subgroup tile and **4 subgroups along the key direction** is fastest, with spill 0 (sg4x2 3353 µs min, sg4x1 3400 µs). 128GRF spills 3.8~11KB in every config (best 5896 µs).
- k32q16: sg8x2 4553 µs (spill 0), sg4x2 spill 64 B; k16q32 spills 21KB even at 256GRF (4sg along the key direction) / 96B (8sg); k32q32 spills 9.6-12KB. → shipped seed = KQ 16×16, sg 4×2 (`sdpa_gen_ocl.cpp:~195`; the limitation is that the MEASURED evidence is a microbench; real-model performance is unverified).
- K-direction alternatives (D=128 t_med µs): **K1 scalar gather 365 (adopted)**, K0 contiguous 286 (assumes alignment), K2 row vload8+pack 370, K3 SLM transpose 449, K4 A=Q/B=K 344 (KQ only, not adopted). V: V0 lane=value gather.
- The micro xe_hpg config (tuning seed): `xehpg_h128 = {16,16,32,8, 8,2, 4,4}` (`sdpa_gen_micro.cpp:322`) satisfies the alpha nesting of sdpa_ocl `solve_sv_split`.
- Failure modes to watch: (1) an SG8 codegen failure is swallowed by `add_stage` and **silently demoted to the opt kernel**; check with a census; (2) PA MIXED f16 fails to compile on DG2 even with `USE_2D_BLOCK_IO_*=0`, because `v_tile_b2d16` always uses the 2D transform (scalar Kc/Vc readers needed);
  (3) offline-checking a source dumped with a faked arch (`OV_GPU_ARCH_OVERRIDE`) on the B70 with ocloc `-device dg2` is possible, but **the execution results are meaningless**.

---

## 12. Procedure for choosing the tiling config of a new kernel

1. **Fix the operation shape and the operand mapping first.**
   - For each GEMM, write down the (lane, component) of A/B/C with the "lane = column" rule. The side that needs VNNI is B (for the OpenCL built-ins), so put the input read as VNNI (V-like) in B, and for an input that has to be read transposed choose A or B according to the axis it can be loaded along.
   - Decide whether the axis being reduced (softmax max/sum etc.) is **the lane or the component**. Lane means shuffle/reduce; component means an in-register sum. If possible, make the reduction axis the **component** (here lane = query).
   - If a layout transpose is needed between the two GEMMs, derive at this point the cost of going through SLM and the invariants (problems like alpha nesting, where "the two stages map subgroups differently").
2. **Encode the invariants in a host solver** (§4). Tile products = coverage conditions, matching subgroup counts, per-subgroup nesting conditions. Whatever can be, as a kernel `#error`; whatever cannot, as a solver unit test. Make the override path go through the same solver.
3. Filter the candidates with a **resource model**: the SLM formula (§3.3), WG ≤ 1024 work-items, a live-register estimate (float8 C count = tile count, `k_raw`/`qB` temporaries). Static prediction (ocloc spill) is **for reference only**; the runtime `SPILL=` is the final verdict (§7.3).
4. **Build rebuild-free knobs first**: expose tiles/GRF/paths as env + a trace output like `SDPA_OCL_TRACE_CONFIG` (confirm with a source dump/trace that the config was actually applied; identical results mean the knob is not on the path). ISA A/B by editing `#define`s in the dumped source + ocloc (→ chapter 04).
5. **Candidate grid**: (total subgroup count ≈ in the thousands) × (wgTK 64/128) × (wgTQ 32/64) × (sg_per_wg 8/16) × (GRF 128/256). Treat **GRF and tiles as a pair** (§7.2). Too few subgroups (≤ 2048) means insufficient TLP, too many (8192) means overhead (gpt-oss: 4096 optimal; ASSUMED, do not generalise, re-measure per shape).
6. **Accuracy first**: time each candidate only after it passes the full accuracy suite (e.g. the 143-test PA). A "fast timing" from a config that violates an invariant is invalid (§4, invariant 3).
7. **Measurement**: B70 clock locked (2800MHz), per-kernel device time (cliloader), min/avg over 3 or more passes, both real models and synthetic cases representative of the shape (head/q/k/GQA/dtype/causal). Do not use SLM/occ% as metrics. Do not promote one shape's result to a general rule (find a discriminator, like `V_TILES`).
8. **Check for regressions on other shapes**: decode/generate, other heads, MIXED, other dtypes (i8/u4), other archs (DG2). Change a default only after checking accuracy + performance + dispatch + memory regressions on every path (AGENTS.md).
9. **Record**: record failed configs too, as (config, device, result, cause hypothesis and whether it was falsified) (§13).

---

## 13. Table of failed/falsified tiling experiments

| Experiment | Result | Cause/lesson | Device · source |
|---|---|---|---|
| `kq_sg_tile_keys` 16→32 (wgTK 128→256) | 5.10M→5.59M ns (+9.5%) | the tile itself is neutral; the derived `kq_wg_tile_keys` and sg_per_wg increase the reduction overhead. "Removing the waste" of the 32-row transform was not the lever | B70, kq-tile-keys-32 |
| "more SLM → lower occupancy" hypothesis | falsified | spill 0, instruction count -2%, fewer syncs, identical dispatch. occ% is inversely correlated with speed | B70, 〃 |
| the k32 candidate with sg_per_wg 16→8 (k32wg) | 4.35M (slower than k16pwg4 3.53M) | at the same sg=8, tile16 is faster than tile32 (but the validity of 6.1 is in doubt) | B70, 〃 |
| overriding only `kq_sg_per_wg_keys` | invalid timings | violates invariant 3: a nonexistent subgroup owns value columns → part of the output not computed | tiling-constraints |
| (kq_tile_q=16, sv_tile_scores=32), (32, 64) | `*paged*96` accuracy FAIL | alpha[] nesting violated | B70 |
| tq64 pwk4 pwq1 (C, sg=4) | 3.89M (1.72x slower than A') | fewer subgroups at wgTQ=64 = too few threads | B70 |
| tq64 pwk8 pwq1 (D, wgTK=128) | 4.19M (1.85x slower than A') | even keeping wgTQ=64, wgTK 128 is the problem | B70 |
| 256GRF applied to the default tiles | 2.18M→2.73M (+25%) | thread count halved | B70 |
| 256GRF @ head 72 (scalar-gather state) | 9.57M→14.12M (+47%) | even with the spill removed, the LSC message count is the cost, and halving the threads weighs more | B70, head72 |
| T128k256 / T128k128 (large tiles, 256GRF) | 4.15M / 13.80M | spill 3.3KB / 14KB. Even with the minimum k0 iteration count (=micro), spill dominates | B70, beats-micro |
| screening spill candidates with ocloc | 4 of 8 spill at runtime | ocloc spill prediction is useless | B70 |
| block-level causal-mask skip (`causal_block_clear`) | instructions +16 (cmp 20→36, sel unchanged) | IGC flattens the branch instead of taking it. The code remains and is harmless (later docs: "always on") | B70 ISA, beats-micro |
| removing the `k_mask` remainder add | instructions -3 (add 60→20, mov 59→82) | IGC replaces it with movs; low ceiling. Not measured on device | ISA |
| decode sg8, M=8 (old default) | head512 623.2M vs 118.3M | M=8 makes q_reg 128GRF = spill 34432 | B70 gemma-4 |
| decode sg4 | 2.2x slower | too few threads | B70 |
| decode 256GRF (head256) | 408.9M→505.3M | threads halved | B70 |
| decode K prefetch | +3.6% slower | KQ already has many independent chains | B70 llama-3.1-8b |
| MIXED B/C (sg8) | 157,877 / 372,628 vs A 146,417 | a narrow key tile + keeping the threads is the key. The spill is invisible to ocloc | B70 llama-3.2-1b |
| u4 MIXED keeping paired Kc DWORDs (long-lived private array) | 177,716 (about 12% worse) | no long-lived private arrays | B70 llama-3.2-1b |
| u4 MIXED KQ_FAST / KQ_TRIM | 172,310 / 157,932 (vs off 151,746 / 151,529) | worse, removed | B70 |
| h(sg8) at gpt-oss (2048 total subgroups) | 427.7M vs 397.9M | too few threads, even with the fewest loads per unit of work | B70 gpt-oss |
| tile_keys=32, pwk≥4, query_blocks≥2 | wrong output (output index 35264 = start of the second query block) | **Cause not found**. Memory-path switches, 256GRF, the tiling invariants, S_slm capacity/conflicts all ruled out. As of 2026-09-25 it does not reproduce on the `MICRO_MATH=1` path (needs re-checking with `MICRO_MATH=0`). The tuned table always uses 16 keys, so it is reachable only by override. **Do not ship with tile_keys=32** | B70, tk32-bug-hunt |
| probe "DROP-IN CONFIRMED" | invalid | compared a state in which both builtins returned 0 (nothing written) | tk32-bug-hunt |
| decode i8, 16 subgroups (head 64 / 128) | 1.26x / 1.29x slower at 4K (0.85-0.89x at 32/1024 prompts) | an i8 workgroup is fast; when the GPU is full 16 subgroups only add waves | B70 lfm2.5 / llama-3.1-8b, sg-m-rule |
| decode u4 S8M4 (head 128, kv group 4) | 304.2 vs S8M2 219.8 at 32/1024 prompts | half the workgroups = lost parallelism at short context (yet the best config at 4K: 208.1) | B70 llama-3.1-8b, sg-m-rule |
| decode u4 S16M2 (phi-4-mini, kv group 3) | 229.5 vs S8M2 195.1 at 4K | without a larger M, 16 subgroups double the threads per partition -> an extra wave; fixed by rounding M up | B70 phi-4-mini, sg-m-rule |
| "16 subgroups make head-128 decode 0.8x" (first reading) | confounded | the register estimate also moved M from 2 to 4 | B70 llama-3.1-8b, sg-m-rule |
| PA prefill head 128 (16, 64, 8, 2) = 128 x 128, 16 SG, 256GRF | 1.20x micro, SPILL 256 | the key tile must also be micro's (256) | B70 qwen3-8b, longctx |
| PA prefill head 64 with micro's tile at REG128 | 2.05x, SPILL 5,952 | the tile needs 256GRF | B70 llama-3.2-1b, longctx |
| PA prefill head 256 (16, 32, 4, 4) / (16, 16, 8, 4) / table + loop | 3.2x (SPILL 14,144) / 0.96x / 1.15x vs shipped 0.774x | | B70 gemma-2-9b, longctx |
| PA prefill d_max 512 (16, 16, 16, 4) at 256GRF | CL_INVALID_WORK_GROUP_SIZE | 1024 work-items over the 256-GRF limit | B70 gemma-4-e2b, longctx |
| PA prefill micro's S*V split (32 values x 16 scores) | +4.7% (qwen3-8b 8K) | | B70, longctx |
| PA prefill epilogue S_sum one load per row | +2% (+6.6% at 2K) | | B70, longctx |
| PA prefill reversed WG order (long query blocks first) | 0% / +0.7% at 32K | | B70, longctx |

Reusable methods from the tk32 debugging: (1) run each env alone and in combination to isolate interactions, (2) force every optional fast path to scalar to separate indexing from memory ops,
(3) decode the wrong output index as (token, head, dim) → tile coordinates, (4) **validate the analytic model on a working config first**, (5) confirm with a sentinel that the probe "really wrote".
Intel subgroup block reads are **strided**: component i of lane L is `p[i*SG + L]` (not lane-contiguous) — this misunderstanding once made an S_slm layout analysis wrong (corrected 2026-09-11; a derived value, not measured on device).

---

## 14. Mismatches between the memory notes and the current code / caveats

- Some notes (head72, decode) cite `sdpa_gen_ocl.cpp:157-164`, but that location no longer holds the comment (stale reference; the fact itself, "256GRF reduces the thread count", survives in the comment in `get_build_options`).
- Config A of `sdpa-ocl-mixed-kc-vc-split` (`d_max ≤ 64`: pwk4/pwq4) is not in the current code (§5).
- `SDPA_OCL_BLOCK_SKIP`, `SDPA_OCL_DKS_ACTIVE`, `SDPA_OCL_PA_CUR_GRAN/SIDE/SV_TRIM/V_PREFETCH`, `SDPA_OCL_MAX_BARRIER_V_PREFETCH` were removed by the 2026-09 refactor (not in the current doc's knob table). Do not treat these env vars from old notes as reusable knobs.
- `kq_sg_tile_keys` currently allows only 16/32, enforced by `#error`.
- `tiling_fits_device()` allows 1024 work-items whatever the GRF mode, but a 256GRF kernel cannot dispatch 1024 (§6.6). Only an env
  override reaches it today (the shipped 256GRF tilings use <= 32 subgroups).
- A note says the `TEST_USE_SDPA_OCL=0 (default)` comments at `sdpa_opt.cpp:~52`, `paged_attention_opt.cpp:~1413` contradict the code (sdpa_ocl is the default). Go by the code's behaviour.
- Every absolute ns refers to the device on that line; B580 numbers are stale until re-measured on the B70.
- §8.1's `get_sg_per_wg` rule and its line numbers are stale; the current rule is in §8.3 (uncommitted as of 2026-10-22). Notes written before 2026-10-22 that call the cache of a 4-bit model "i8" are wrong: it is u4 by default.
