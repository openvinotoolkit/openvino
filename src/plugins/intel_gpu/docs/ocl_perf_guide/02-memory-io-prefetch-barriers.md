# 02. Memory IO, prefetch, barriers (Xe2 / Xe-HPG, sdpa_ocl family)

Scope: memory-path techniques actually used or tried in the `sdpa_ocl*.cl` kernel family (`src/plugins/intel_gpu/src/graph/impls/ocl_v2/`).
For DPAS/tiling see `01-dpas-and-tiling.md`, for numerics/quantization `03-numerics-softmax-quantization.md`, for ISA/spill/profiling
`04-spill-isa-profiling.md`, for methodology `05-methodology-and-pitfalls.md`.

## 0. Notation and evidence grades

- Hardware: **B580** = Arc B580 (Xe2, early measurements), **B70** = Arc Pro B70 (Xe2, dev machine since 2026-07-05), **DG2** = Arc A770 (Xe-HPG, SG8),
  **iGPU** = GPU.0 of this PC (no XMX). B580 and B70 are the same Xe2, so builtin availability/layout/ISA analysis stays valid, but absolute ns are device-dependent.
- **MEASURED** = measurement run by the user (source given). **ISA** = static ocloc/IGC ISA count (not runtime time; in this project it disagreed with time several times, §13).
  **ASSUMED** = hypothesis, not measured. AGENTS.md rule: never claim a performance improvement without measurements.
- Code citations are `path:line` relative to `ocl_v2/` (current working tree; line numbers drift with edits).
- Dates of memory notes are given by the note file name (`~/.claude/projects/.../memory/<name>.md`).

## 1. 2D block IO builtin summary

Extension: `cl_intel_subgroup_2d_block_io` (`#pragma OPENCL EXTENSION ... : enable`). Available only on Xe2 (bmg).
**DG2/ARL-H (xe_hpg) do not advertise this extension and reject read/write/prefetch all at compile time** (MEASURED, DG2 S0 probe, `sdpa-ocl-xe-hpg-facts`; compute-runtime
`compiler_product_helper_before_xe_hpc.inl`). So the SG8 arm turns 2D block off with a jit switch: `block2d_io_allowed()` (`sdpa/sdpa_gen_ocl.cpp:720`).
On DG2 the pragma only emits a warning and the macro is not defined, so it cannot be switched off with `#ifdef`.

Signature shape (read): `intel_sub_group_2d_block_read_<variant>_<bits>_<rows>r<cols>x<count>c(void* base, int width_bytes, int height, int pitch_bytes, int2 coord, T* dst)`.
The unit of `coord.x` is **elements** (byte for 8b, half for 16b, dword for 32b), `coord.y` is rows. width/pitch are in **bytes**.

| Use | builtin | lane meaning | Used in |
|---|---|---|---|
| plain read | `read_16b_16r16x1c` | lane=column, element=row(16) | KQ A operand for f16 K input/page (`sdpa_ocl_qk_load.cl:256,330`) |
| plain read 32b | `read_32b_8r16x1c` | lane=dword column, element=row(8) | u4 Kc as a DWORD surface (`sdpa_ocl_qk_load.cl:466`) |
| transpose | `read_transpose_32b_16r8x1c` | lane=row, 8 dwords = the 16 halves of one row | Q B operand (`sdpa_ocl_qk_load.cl:24`), decode K B operand (`sdpa_ocl_decode.cl:547`), f16 d-major K page at 128 GRF (`pa_k_tile_dt`, §4.5) |
| transpose 32-row | `read_transpose_32b_32r8x1c` | lane L = rows y+L **and** y+16+L, interleaved: element 2i = (y+L, x+i), 2i+1 = (y+16+L, x+i) (probed on the B70, `test/slower/probe/probe_t32.cpp`) | f16 d-major K page at 256 GRF: depth tiles db and db+2 in one read (§4.5) |
| transform(VNNI) 16b | `read_transform_16b_16r16x1c` | lane=column, dword = 2 rows | V B operand (`sdpa_ocl_v_load.cl:55`), decode V |
| transform 8b | `read_transform_8b_32r16x1c` (`_32r16x2c`, `_32r16x4c`) | lane=column, uint = 4 rows as bytes | i8/u4 K·V (`sdpa_ocl_qk_load.cl:275`, `sdpa_ocl_v_load.cl:87-99`) |
| write | `write_16b_8r16x1c` | lane=column | O output (`sdpa_ocl.cl:1033`) |
| prefetch | `prefetch_16b_16r16x1c`, `prefetch_8b_32r16x1c`, `prefetch_32b_16r8x1c` | no destination (null destination) | `sdpa_ocl_v_load.cl:30`, decode `sdpa_ocl_decode.cl:190,197` |

builtin availability (MEASURED, ocloc `-device bmg`, `sdpa-ocl-8b-transform-32row-min` 2026-07-08):

| Kind | Exists | Missing |
|---|---|---|
| 8b transform | `32r16x1c`, `32r16x2c`, `32r16x4c` | `16r16x1c/2c/4c` (**no 16-row transform**) |
| 8b plain | `16r16x4c`, `16r32x1c`, `32r16x4c` | `32r16x1c` (1-byte needs a column count that is a multiple of 4, except transform) |
| transpose | 32b only (`transpose_32b_16r8x1c`, `_32r8x1c`) | `transpose_8b_*`, `transpose_16b_*`; IGC 2.41's libigc also carries `transpose_32b_16r16x1c` and `transpose_64b_*`, but OpenCL C does not declare them ("undeclared identifier", ocloc, 2026-10-27) |
| 16b plain | `16r16x1c`, `32r16x1c`(compiles, layout unverified, `sdpa-ocl-tk32-bug-hunt`) | |
| prefetch 8b | `32r16x1c`, `16r16x4c`, `16r32x1c/2c`, `32r16x2c/4c`, `32r32x1c/2c`, 1/2/4/8 rows | (prefetch has no transform/transpose distinction) |

Always **confirm the lane/element meaning with a device probe** (`test/microbench/probe_v_layouts.cpp`, `probe_k_16b_32r.cpp`, `probe_v_multiblock.cpp`,
`run_probe.sh`). The evidence is back-tracing (key, head) from the values read (fill K[key][h]=key and print which key each slot holds).
Trap: a probe once reported "match" while both builtins had written nothing (all-zero comparison). Always fill with a sentinel and check the "number of slots written"
(`sdpa-ocl-tk32-bug-hunt`, lesson 5). **Whether `16b_32r16x1c` is a drop-in for `2x16b_16r16x1c` is not verified yet** (the probe was invalid).

## 2. Rules: spec vs our helper

Compares the undefined-behavior (UB) conditions of the spec (`cl_intel_subgroup_2d_block_io` §6.13.X.6, looked up 2026-08-11, `sdpa-ocl-8b-transform-32row-min`), the rules our host helper
(`sdpa/sdpa_ocl_utils.hpp`) applies, and the boundaries actually measured on B70.

| Rule | spec | our helper | B70 measured |
|---|---|---|---|
| subgroup | size 16, full | `block2d_io_allowed(sg==16)` (`sdpa_gen_ocl.cpp:720`) | - |
| `coord.x` | multiple of 4 for 8b, multiple of 2 for 16b | guaranteed in the kernel (`sg_j0_sv`, `db*DPAS_K`, etc.) | - |
| base address | 64 B aligned | strict tier `%64`; fixup tier corrects in-kernel | **a base misaligned by 4/16/32/48 B reads correctly even without fixup if the stride is a multiple of 16 B**, misaligned by 2 B gives wrong results (`sdpa-ocl-ki-t7`) |
| width | 64..224 B, multiple of 4 (8/16b) | `row_bytes >= 64` | **the 224 B upper bound works even when actually violated**: f16 K read is correct with width 256 at head 128 and 1024 at head 512 (B70). Relying on it is forbidden |
| pitch | >= width, multiple of 16 B | `row_bytes % 16` (fixup) / `% 64` (strict) | token stride 260 B/264 B (not a multiple of 16) gives **wrong results** (0.0118~0.068 error) |
| height | (0, 224] | page read clamps (`kp_rows = PA_PAGE_ROWS(k, key0)`) | HW fills OOB rows with 0 |

`block2d_surface_ok(row)` = `row >= 64 && row % 64 == 0` (`sdpa_ocl_utils.hpp:52`): **stricter** than the spec. The spec effectively requires only `width >= 64` and `pitch % 16`.
i8 pages of head 80/96/112 (row 80/96/112 B) are legal per spec but our helper excludes them (`sdpa-ocl-8b-transform-32row-min`).

### 2.1 Block IO legality checklist

All must pass before adding a new 2D block path or changing a gate.

| # | Check | How to verify | Symptom if violated |
|---|---|---|---|
| 1 | Is the subgroup size 16 (on DG2 SG8 the builtins do not exist at all) | host `block2d_io_allowed` | DG2 compile failure or silent wrong result |
| 2 | Does the (bits, rows, cols, variant) combination exist | `test/microbench/probe_dpas_api.sh`, ocloc | compile failure |
| 3 | For 8b transform, do you accept 32-row consumption/over-read; for transpose, is it 32b | §4 | no 16-row variant |
| 4 | width >= 64 B, multiple of 4 for 8b/16b | row_bytes computation | UB |
| 5 | Is the pitch a multiple of 16 B (for rank-2 PA, the token stride including padding) | §3.5 | **silent wrong result** (B70 measured) |
| 6 | Did you prove 64 B base alignment, or enable fixup (round-down, x shift, width widening) | §3.2 | wrong result or (on B70) correct by chance |
| 7 | For a 32b transpose fixup, is `prem % 4 == 0` provable (Q, A) | Q/A currently have no fixup at all | WRONG result, not a performance problem |
| 8 | Is the unit of `coord.x` right (8b=byte, 16b=half, 32b=dword) | code review | columns off by 1/2x |
| 9 | Did you clamp the surface height to the actually valid rows (unwritten page slots, last k0 tile) | `kp_rows`, `y_sub` | NaN from unwritten slots leaks into the scores |
| 10 | Do you avoid issuing reads with height <= 0 | `if (kp_rows > 0)` / zero-fill | illegal read |
| 11 | Is the head tail (`d` not a multiple of `DPAS_K`) left to HW zero-fill via the surface width (`d*sizeof`), or is there a kernel guard | `KP_w = d * sizeof(half)` | values of the next head mix in |
| 12 | For dynamic padding, did you acknowledge that the host cannot prove it and document the assumptions (stride %16, start %4) | `block2d_layout_fixup_ok` comment | dynamic K/V wrong result (residual risk) |
| 13 | Does the assumption that the block read target is adjacent rows continuing into the next read (paired read) hold for PA pages | pages are non-adjacent | reads a different page |
| 14 | Is a forced `SDPA_OCL_*_2D=1` override safe (is the fixup flag derived after the override) | §3.2 | timing right, result wrong (12/16 heads wrong at head 72) |

## 3. Host gates: tiers, base fixup, padding

### 3.1 Tiers (current code)

| Tier | Function | Condition | Target |
|---|---|---|---|
| strict | `block2d_layout_ok(layout, row)` (`sdpa_ocl_utils.hpp:112`) | `row >= 64 && row % 64 == 0` + padding proof | Q, A (output), plain i8 K/V, base without fixup |
| fixup | `block2d_layout_fixup_ok` | `row >= 64 && row % 16 == 0` + padding proof | plain f16 K/V, MIXED Kc/Vc, turns on `BLOCK2D_KV_BASE_FIXUP`/`CUR_BASE_FIXUP` |
| page | `block2d_page_ok(row)` | `row >= 64 && row % 16 == 0` | PA cache pages (f16, i8 BY_TOKEN/BY_CHANNEL) |
| u4 | `block2d_surface_ok` (strict) | keeps `%64` | u4 K/V pages, mutually exclusive with 1D uc16 reads (§5.2) |
| decode | `block2d_surface_ok` (`sdpa_gen_ocl_decode.cpp:49-50`) | strict `%64` | decode K/V pages: f16 `h%32`, i8 `h%64`, u4 K `h%128` |

Host call sites: `sdpa_gen_ocl.cpp:733-744` (Q, KV, A), `:792-823` (PA pages), `:835-848` (Kc/Vc), `:857-868` (plain i8).

### 3.2 `%64 -> %16` relaxation and base fixup (gemma-4 head-72 prefill)

**What**: `block2d_surface_ok()` **lumped three rules (width/pitch/base) into `row_bytes % 64`.** For head 72 f16 (row 144 B), USE_2D_BLOCK_IO_Q/KV/A all became 0 and
K/V fell back to **256 per-lane scalar gathers** per subgroup per k0 (micro: about 13 block reads). Checking the rules separately, the only violation was base alignment
(`base + head*144`; width 144 B and pitch 2304 B pass).

**Fix (2 independent levers, both landed, `191722ba0e`)**:
1. Relax the gate to `row_bytes % 16` (pitch and base are integer multiples of row, so `%16` implies width/pitch and only base remains). The remaining base is corrected in-kernel
   (the way micro `block2d_load` always did it): round base down to 64 B, `x += prem/elem`, `w += prem`.

```c
// sdpa_ocl.cl:300-309 (BLOCK2D_KV_BASE_FIXUP)
const uint k_prem = (uint)(as_long(K) & 63);
K_b2d   = (const global KEY_DATA_T *)((const global char *)K - k_prem);
KD_w_b2d = KD_w + (int)k_prem;
KD_x0   = (int)(k_prem / sizeof(KEY_DATA_T));
// read: at (int2)(KD_x0 + db*DPAS_K, key_base) with width KD_w_b2d
```

   The widening extends **backwards (to lower addresses)**, so columns past the head dim stay OOB and the HW fills them with 0. Same effect as the existing `head < d` guard; values of the next head do not leak in.
   That `base - prem` does not go below the allocation is proven by `base = buf_base + m*144` with `buf_base` 64 B aligned (`sdpa-ocl-block2d-padded-view-gate`). Cost 0
   (2 ANDs hoisted out of the loop, MEASURED within 0.7%, B70/B580 `sdpa_ocl_head72_analysis.md`).
   → head-72 prefill 6.96x (MEASURED, same document).
2. `DKS_ACTIVE = ceil(HEAD_SIZE/DPAS_K)` (5 vs 8 at head 72; 3 tiles loaded only 0s and fed 0s to DPAS). → a further 1.33x; it exceeds the work reduction (-19%) because
   `Q_slm` 8192->5120 takes WG SLM 17536->14464 B, so **resident WGs per Xe2 Xe-core (128 KB) go 7->9** (MEASURED interpretation, not ISA; occupancy arguments have
   missed twice before in this project, so treat it as ASSUMED).

Overall: gemma-4-26b-a4b-it SigLIP prefill (head 72, 16 heads, q=k~2528) 9,570,971 -> 1,030,869 ns avg (**9.28x, 1.14x faster than micro**) (MEASURED, B70, cliloader,
`sdpa-ocl-block2d-gate-relaxation`). Correctness: `*ScaledAttn*-*486*:*387*` all PASS (`--device_suffix=1`).

**The fixup flag is derived after the override** (`sdpa_gen_ocl.cpp:741`: `BLOCK2D_KV_BASE_FIXUP = kv_2d && !kv_aligned`). Before that, forcing it with `SDPA_OCL_KV_2D=1`
made 12/16 heads wrong at head 72 (only the timing was right). Now this override is safe as an "A/B lever that proves a hypothesis without a rebuild".

**Side results (contrary to three intuitions; do not re-derive)**:
- **256 GRF was actually 47% slower** (9.57 -> 14.12 ms) even though it removed the spill. The bottleneck is the number of LSC messages, not spill, and halving the resident threads cost more (256GRF comment in `get_build_options`, `sdpa_gen_ocl.cpp:1038-1041`).
- SPILL=2688 was a symptom, not the cause: it incidentally became 0 once the load method changed.
- `Q_2D`/`A_2D` are perf-irrelevant (Q staging is in the prologue, the A store in the epilogue, outside the k0 loop of about 20 iterations): `SDPA_OCL_Q_2D=1` measured +0.35%. Q/A have no base fixup, so forcing them risks **wrong results**.

### 3.3 padded view gate (`bd8d90097d`, phi-4 vision tower)

The fixup tier required `!l.data_padding`, so the fused-QKV `Split` crop view (in-place padded) stayed on scalar gather. The phi-4-multimodal vision tower (head 72, 16 heads,
q=k=1024, mask kind 2) was **7.5x slower** than micro. B70 measurement (208 calls): micro b=2 411,976 ns, ocl before 3,097,727 ns (7.52x slower), `SDPA_OCL_KV_2D=1` 393,355 ns (1.047x faster);
b=5 micro 1,022,198 / ocl before 8,769,732 / 971,180. SPILL 9,344 B -> 0 (MEASURED, `sdpa-ocl-block2d-padded-view-gate`).

Core logic: in a simple data format every OV pitch is a product of padded dimensions = an integer multiple of `X_PITCH` (= padded_x). **If X has no padding**, `row%16` implies `pitch%16` and `>=64`.
batch/head/seq padding only **multiplies** the pitch (`pitch = (16+padY)*72*2` is always a multiple of 144). pad-before folds into `INPUT*_OFFSET` as an integer multiple of rows, so base is off by a multiple of 16 and
fixup recovers it exactly. Implementation: `axis_unpadded(layout, ChannelName::X)` (`sdpa_ocl_utils.hpp:69`) reads `padding::_lower_size/_upper_size/_dynamic_dims_mask` via `get_channel_index`
the same way as `jitter.cpp:90-92`. Fallback cost (per subgroup/k0): K 5*2*8 = **80 gathers vs 5 block reads**, V 8*16 = **128 vs 8** (16x the message count); about 16 live gather addresses cause the spill.

### 3.4 rank-2 PA padding (T7, `1fece9b9c1`)

PA Q/K/V are rank-2 `[tokens, heads*head_size]`. The head dimension lives inside FEATURE, so feature padding moves the base (by the padding-before amount) and every head's token stride by an **arbitrary amount**.
The rank-4 argument "pitch and base are integer multiples of row" does not hold, and `axis_unpadded(X)` is vacuously true (the format has no X -> `get_channel_index` = -1).

| Input | strict | fixup |
|---|---|---|
| rank-4 | simple format && X unpadded | same |
| rank-2 static padding | `first_head % 64 == 0 && token_stride % 64 == 0` | `first_head % 16 == 0 && token_stride % 16 == 0` (more conservative than the device) |
| rank-2 dynamic padding | **impossible (false)**: Q/A go scalar, K/V/Kc/Vc go to the fixup tier | **trusted (true)**: assumes `stride % 16`, first head start `% 4` |
| other ranks | rejected | rejected |

B70 measurement (`paged_attention_feature_pad_test`, 55 cases, `sdpa-ocl-ki-t7-block2d-padding`): **the reproducible defects are a token stride that is not a multiple of 16 B (260/264 B) and a first head start misaligned by 2 B**.
Even with the base misaligned by 4/16/32/48 B, a stride that is a multiple of 16 B reads correctly without fixup - my prior prediction ("a 64 B base breaks") was wrong. The gate still keeps the documented 64 B/16 B rules
(spec + an earlier B580 phi-4 experiment). Therefore **the base rule is guarded only by the host test `sdpa_block2d_gate`; the device test proves only the pitch.**
Residual (not fixable on the host): dynamic K/V padding with stride %16 != 0 or start %4 != 0 -> wrong result. Repro: `--gtest_also_run_disabled_tests --gtest_filter=*paged_attention_feature_pad_residual_test*` (4 FAIL).
Evidence: pset rT8->rT7 changed `USE_2D_BLOCK_IO_Q 1->0` in exactly 105 configs (pa-mixed 75 + pa-prefill 30, all DYNAMIC_INPUT_PAD), 0 other changes, `.inc` identical (host-only).

**Data trap (MEASURED)**: with the PA harness data N(0,0.1), q.k ~0.08, so softmax is almost uniform and **Q/K misreads are hidden** (only V misreads show). Adding `logit_scale_gain` (constant scale x128,
sharp softmax) was needed to expose the Q misread (case 47: 0.068 vs non-sharp 13: 0.003). Even then not always (case 30: 0.0021). Details in `05-methodology-and-pitfalls.md`.

### 3.5 pitch unit trap (a K page rule independent of head size)

If the K page is d-major `[head_size, block_size]`, row = `block_size` tokens = 16 * 2 = **32 B**, below the 64 B width and also violating the `pitch >= width` rule. Since the row length is block_size, not head_size,
no head size avoids this (`pa-k-cache-layout-d-major`). This is the root reason for the token-major relayout in §9.

## 4. 8b transform is 32-row only, transpose is 32b only

### 4.1 Facts

- The 8b VNNI transform read exists on bmg **only in 32-row form** (§1 table). K/V pages are 16 tokens, so a 16-row call is impossible.
- Result: in the default head-128 config (`kq_sg_tile_keys=16`, cp step=16), the K·Q read and the S·V read fetch 32 keys but consume only the first 16 (uint 0..3) -> **50% row over-read** (decoded but discarded).
- **But attempts to cut the read COUNT did not reduce time** (§12 table: all 3 cp-pair reuse attempts failed). The bottleneck was the dequant mov storm, which was solved by the bias trick.
  So "removing over-read" only matters if (a) a 16-row, VNNI-aligned, shuffle-free path exists and (b) it actually reduces device time - **neither is proven**.
- micro has **0 over-read** (gemmstone source trace, `sdpa-ocl-8b-transform-32row-min`): nGEN programs the block2D descriptor height directly, so it is not bound to the fixed 32-row geometry of the OpenCL builtin.
  micro's A operand is decided by `problem.A.layout`: T -> Block2DTranspose(K, `ka_load=16`, d32 transposed read that reinterprets s8 as u32), N -> Block2DVNNI(V, crosspack=4).
  OpenCL does not expose descriptor control, so this cannot be mimicked exactly.

### 4.2 When transpose is 32b only: view the i8 surface as dwords

`read_transpose_*` exists only for 32b. To transpose-read an i8 K page, **give the same `transpose_32b_16r8x1c` call a width/pitch in bytes**
(4 bytes = 4 consecutive elements form one dword). Legal if the pitch is a multiple of 16 B. 8 dwords = 32 bytes = 32 head dims = **2 DPAS tiles**, so the K block messages are halved (head 128: 16 -> 8)
(`sdpa-ocl-decode-kernel`; ISA count). The `x = r*8` dword coordinate is the same for f16/i8/u4, so the K read and the K prefetch share one call site.

```c
// sdpa_ocl_decode.cl:547 - common to all precisions (f16: 8 dword=16 half, i8: 32 byte, u4: 64 nibble)
intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void*)(key_cache + k_page_off[g]),
        K_ROW_BYTES, PAGED_ATTENTION_BLOCK_SIZE, K_ROW_BYTES, (int2)(r * 8, 0), (private uint*)&kt);
```

### 4.3 u4 and Kc: channel pairs as dwords (and the DPAS depth permutation)

One byte of a u4 K page is a channel pair (2b, 2b+1), so an 8b transform read gives lane L the channels (2L, 2L+1), not the contiguous `base+L` that DPAS wants. Depth is the reduction axis, so using the same
permutation for A and B preserves the result (`PA_K_U4_CHANNEL(db,L) = win + 2L + (db&1)`; `docs/sdpa_ocl.md` "u4 K: the permuted depth axis"). To read Kc (raw f16 current token) in the same order,
**Kc is viewed as a DWORD surface**: `read_32b_8r16x1c`, x in dwords (`sdpa_ocl_qk_load.cl:466`). Each window is read twice, once per parity (L1 hit), but with about 4x fewer instructions than the page dequant.
Checked at runtime with `kc_dword_ok = ((KcD_x0 & 1) == 0)`, falling back to per-lane gather on failure (`sdpa_ocl.cl:284`). Details in `03-numerics-softmax-quantization.md` (dequant).

### 4.4 Block-reading a d-major i8 page through a reinterpreted surface (NOT hw-verified)

A design dropped on 2026-08-12 when the user chose token-major while reviewing BY_CHANNEL, recorded because it refutes the claim "d-major cannot be block-read". The page is contiguous, so a **different surface** can be declared:
in BY_CHANNEL `[k_head_size cols][block_size+4]` (20 B per column), with width = pitch = 80 B (4 columns) and height = `k_head_size/4`, reinterpreted row r is head dims 4r..4r+3, and
`transform_8b_32r16x1c` at `coord.x = 20*c` reads only the 16 bytes of head dim 4r+c (inline comp bytes 16..19 are outside the window). It satisfies every spec rule (width 80, pitch %16, x=20c %4, base %64).
**Trap**: the depth axis is permuted (slot i = head dim 4i+c) -> Q and the per-channel scale/zp must be read in the same order. **The reinterpreted surface read itself has never been run on hardware**; a small probe is needed.
The same idea for the f16 d-major page was implemented, probed and shipped on 2026-10-27 (§4.5).

### 4.5 Block-reading the d-major f16 K page in MIXED (`USE_2D_BLOCK_IO_K_PA_DT`, 2026-10-27, B70, MEASURED)

The f16 K page `[head_size][16 tokens]` has a 32 B row, so MIXED gathered K one key per message (qwen3.5-9b 8K: 1,465 us per
call, `sdpa_micro` 522). Instead of the layout change (§9, token-major: 441 us, but every page reader moves), the page is
re-declared as a **64 B pitch surface** (width = pitch = 64 B, height = `head_size / 2`): row r = channel 2r (16 keys = 8
dwords of key pairs) followed by channel 2r+1. A **32b transpose** read gives lane L surface row y+L, element i = dword column
x+i; at x = 8 * (db & 1), y = 16 * (db >> 1), lane L holds channel 32 * (db >> 1) + 2L + (db & 1) = the u4 permuted depth
(§4.3, `PA_K_U4_CHANNEL`), so the u4 Q staging and Kc dword read serve it unchanged (`PA_K_PERMUTED`). The k_raw rows are
the word halves of the dwords (one `and` per row, see the mask below). Code: `pa_k_tile_dt()` in `sdpa_ocl_qk_load.cl`,
gate `pa_k_dt_jit()` (f16/bf16 d-major page, 16-token pages, `K_HEAD_SIZE % 32 == 0`).

| Step (qwen3.5-9b `d_max` 256, 8K, us per call) | us | Note |
|---|---|---|
| per-key gather (before) | 1,465 | 256 gather messages per key tile |
| 16-row transpose per depth tile, branch for the partial page | 565 | 16 phi `mov`s per depth tile after the branch merge |
| same, partial page masked by `and` with `sub_group_broadcast(key_ok, krel)` (no branch) | 508 | read + 16 `and` per depth tile = token-major's read + 16 copies |
| 32-row transpose (`32r8x1c`): tiles db and db+2 per message | 460 | VTune before it: Send stall 7.8% vs 2.4% (token-major), L3 miss 26% vs 22% |
| + new tokens from the page, not Kc (f16 only, `docs/sdpa_ocl.md` "Paged-attention MIXED: i8 and f16 caches") | 406 | token-major without Kc: 401 |

- The 32-row form keeps 16 dwords per parity live across the depth loop: at 256 GRF no spill, at **128 GRF 4 KB spill** (d_max 64
  table tiling, llama-3.2-1b 497 vs 413 us with the 16-row form) -> `PA_K_DT_ROWS` = 32 with 256 GRF, 16 with 128.
- Unwritten slots of the last page can hold NaN; the read cannot clamp keys (they are columns), so each extracted row is ANDed
  with its key's lane of an all-ones-below-`kp_rows` mask (a scalar region operand: the extraction `mov` becomes an `and`, free).
  A fully-OOB key group reads page 0 (`pa_k_pages()` clamps) and is masked whole.
- Checks: sharp-softmax rows `basic/180`-`185` (head 64/96/128/256/512), reading the other parity on purpose fails all 9 f16
  MIXED rows; the depth permutation reorders the KQ sum (order-changing, not bit-preserving by construction).

## 5. Classic block read, 1D page read, local block IO

### 5.1 Classic subgroup block read

- `intel_sub_group_block_read_us8`, `intel_sub_group_block_read8` (pA from SLM `S_slm`, `sdpa_ocl.cl:916-920`), `intel_sub_group_block_read(const global uint*)` (SG8 K tile, per-channel comp load `sdpa_ocl_qk_load.cl:181`).
- 4 B address alignment is required. **DG2 measured UB**: `block_read uint` with an odd row pitch gives wrong results (err 49), and `block_read_us` at a 2 B offset is also wrong (`sdpa-ocl-xe-hpg-facts`, S2). The contiguous K reader uses block read only when "pitch even + 4 B aligned", otherwise it falls back.
  Actual code: `dword_ok && whole_tile` in `k_tile_dword` (`sdpa_ocl_qk_load.cl:479-522`); the fallback does a dword load aligned with `addr & ~3`, then shifts. (A fallback that loads 2 `ushort`s has a history of compiling query block 1 wrong on DG2.)

### 5.2 u4 head-64 1D whole-page read (`intel_sub_group_block_read_uc16`)

Problem: for u4 head 64, K row = 32 B and V row = `Align(32,16)` = 32 B, both below the block2d 64 B minimum -> both K and V are per-lane byte gathers. Per k0 **K 64 + V 128 = 192 `load.ugm.d8u32` vs 16 dpas** (ISA),
+ 5952 B spill. gpt-oss-20b MIXED was 3.67x slower than micro (not the originally reported sink cause: 13/3529 instructions).

Fix: the data area of the page is `16 * ROW` contiguous bytes, and `block_read_uc16` puts component i at byte `SUBGROUP_SIZE*i + L` of lane L. With `COLS = ROW/16`

```
PA_PAGE_UC16 * r + i == t * COLS + c          (r = read index, i = component, t = token, c = 16 B column group)
```

No shuffle and no per-lane address (`sdpa_ocl_config.cl:340-362`, macros `PA_PAGE_R/I`, reads `sdpa_ocl_qk_load.cl:236`, `sdpa_ocl_v_load.cl:291`).

- `r` is a compile-time constant even when c is a runtime value, because `COLS` is a power of 2. `i` is not, so **V biases the base by `SUBGROUP_SIZE*c` and passes c=0** (a runtime `i` turns the `uchar16` subscript into indirect register addressing).
  K's c is `db>>1` (a constant), so it reads from the plain page base and the 2 reads are hoisted out of the `db` loop (that hoist is most of the gain: 64 -> 2).
- ISA (bmg): `uc16` -> 1 `load.ugm.d32x64t.a64` (256 B), `uc8` -> `d32x32t`. Also works on `__local` (`load.slm.d32x64t`). The last byte accessed is at 527 B for head-64 u4 V, within the page (576 B).
- Guards intentionally removed: keys `>= k` get -INF added to their score, so the load only has to guarantee **finiteness** (u4 nibbles are in [0,15]). Removing the per-key `key < k` cut cmp 240 -> 39. `head < d` is folded into the scale.
- Static result (head 64): instructions 5733 -> **3516**, d8u32 gather 192 -> 0, spill 138 -> 0. This path is enabled only where block2d is off (u4 head 32/64, V only for head 48). When relaxing, u4 keeps strict `%64`: the 1D read is gated on "block2d off", so loosening it to `%16` would replace the 3.92x path whose gain is proven.
- **MEASURED (B70, gpt-oss-20b whole run)**: 1474574702 -> 632343047 (+1D) -> 397934594 (+`KQ_TILE_QUERIES=32`, already faster than micro 401436373) -> **376221254** (+V uniform-shift nibble) = **3.92x**, 6.3% faster than micro (`sdpa-ocl-u4-head64-page-read`).
  Note: a total subgroup count of about 4096 is the sweet spot (8192: 632 ms, 4096: 398 ms, 2048: 428 ms), and the "minimum message count" (config h) actually lost -> the static message count is the wrong metric for this kernel.
- Sharing the 4x redundant V dequant through SLM (about -24% ops) was **dropped** because SLM +16 KB -> WG 4 -> 3 (-25% occupancy) (a blind bet). Occupancy has already been wrong twice.
- Not applied: `sdpa_ocl_decode` has the same defect (128 `load.ugm.d8u32` in the gpt-oss dump). The uc16 mapping applies as is, but the decode gap was never measured, so it is out of scope.

### 5.3 local (SLM) block IO and xe_hpg (DG2)

DG2 S0/S2 (MEASURED, compile+ISA + 36/36 mini-SDPA on real hardware): local uint/ushort block IO **succeeds without a pragma**, global uint/ushort block IO succeeds, and the SG8 arm that reads pA from `S_slm` with
`as_int8(intel_sub_group_block_read8((local uint*)&S_slm[...]))` works (`sdpa_ocl.cl:911-914`). DG2 rejects both 2D block IO and subgroup buffer prefetch, and generic `prefetch()` adds
1 null-destination `load.ugm`, but its effect is unmeasured. On DG2 an SG16 `short8` DPAS operand **drops the DPAS without an error** (no dpas in the ISA): a silent wrong-result risk, so a `dpas` column of 0 in the `hpg` subcommand counts as a failure.

### 5.4 Wide reads in the PA PREFILL loop (`WIDE_READS`, IGC 2.41, B70)

Full tables: `docs/sdpa_ocl.md` "Paged-attention PREFILL: wide reads (IGC 2.41)". Three independent bits, each value-preserving
(same bytes in the same registers, same DPAS order; 173 `basic` PA rows bit-identical with every mask):

| Bit | Read | Narrow form | ISA effect per key tile (IGC ShaderDump) |
|---|---|---|---|
| 1 K pair | one `16b_32r16x2c` (32-key SG tile, 2 KB) or `16b_16r16x2c` (16-key, 1 KB) per depth-tile pair, block-major into `k_wide[]` | one `16r16x1c` per depth tile | block2d K reads halved (d128 8 -> 4); nothing else moves |
| 2 V pair | one `transform_16b_16r16x2c` per value-block pair (needs `sv_value_blocks` even) | one transform per value block | V reads halved; inactive where a subgroup holds 16 values (`d_max` 64 / 128) |
| 4 Q_slm | `intel_sub_group_block_read_ul4` (`load.slm.d64x64t`, 512 B), staged by per-lane `ulong` stores | `block_read8` uint = 2 x `d32x64t` | Q SLM messages halved, but **8 `mov (16) r<1>:d r<2;1,0>:d` per read** (ulong4 -> int8 DPAS operand): +256 movs per key tile at d128, +128 at d64 |

MEASURED (1800 MHz, cliloader per call, two rounds): defaults `d_max` 512 = 7 (-21..-31% at 8K-32K, the kernel was Send-bound:
VTune Send stall 26.5% vs micro 5.1%), 64 = 5 (llama-3.2-1b -1.4..-5.6% at 2K-32K) but 0 for a layer with a sliding window,
128 / 256 = 1 (-0.1..-1.0% / -1..-5%). The Q_slm bit is **regime-dependent**: it pays only while the loop waits on memory, so
its gain grows with the key count (llama-3.2-1b alone: +2.4% at 1K, -2.9% at 32K) and turns into a loss where K/V stay in
cache: short prompts, d128 (+1.4..+4.0% alone, +3.8..+16% with K), and gpt-oss-20b's 128-key window layers (K + Q_slm
+10..+12% at every length). The bits do not add up, and the K pair's sign follows the kernel variant: gpt-oss-20b (sink
input) full layers were +1.2..+1.8% slower with the K pair alone at `d_max` 64 while llama-3.2-1b was -0.4..-0.7%. So a
per-tiling default must be checked per variant (window, sink, image, short prompts) and split per kernel hash.

- **Trap (correctness)**: `intel_sub_group_block_write_ul4` to SLM stored Q wrong on the dynamically padded head-64 PA rows
  (`basic/29`-`34`) with `WIDE_READS=7`, even with `Q_slm` 64 B aligned; per-lane `ulong` stores of the same layout pass and
  cost nothing measurable (staging runs once per workgroup). Prove the new path with the bit **forced on every head size**.
- A movs-free 512 B Q read would need inline vISA (`lsc_load.slm d64x64t` into a dword-typed destination): chapter 08 #8 saw
  the inline-asm verifier reject `d64x64t` on DG2 ("more than 8 registers" at 32 B GRF); on Xe2 (8 x 64 B GRF) it is untested.
- Chunked prefill: with GenAI's continuous-batching default scheduler (256-token chunks) only the first chunk runs PREFILL;
  the rest runs MIXED, which has none of these reads, so PREFILL read tuning does not show there.

## 6. Using SLM vs not using it

| Item | Content | Evidence |
|---|---|---|
| sdpa_ocl SLM composition | `Q_slm` (staging Q as the DPAS B operand), `S_slm` (feeding the softmax result back as pA), `S_sum_slm`, `S_max_slm` (local fmax atomic) | `sdpa_ocl.cl:301-304` |
| micro SLM | all of it is the wrapper's Q/S staging (107,008 B vs ocl 17,536 B, 6.1x). **The spliced GEMM (`ugemm_kq/vs_slm_size`) uses 0 SLM**: K/V stream from global straight into the systolic array (prefetch only) | `sdpa-ocl-slm-vs-micro` (ISA/source formula, MEASURED diff 0) |
| Cause of the difference | only the two tile sizes: query tile 128 vs 32, key tile 256 vs 128. Not a difference in implementation approach | same note |
| Occupancy | Xe2 Xe-core SLM 128 KB. micro 1 WG/Xe-core (82%), ocl 7 WGs. micro still wins via causal efficiency (80% vs 50%) + 256GRF -> **occupancy headroom is not throughput** | same note |
| decode S*V | head-dim split takes `slm_out` (16,640 B) -> `slm_p` (2,304 B), -86%: removes the 7/8 WG per Xe-core occupancy cap. The time effect of this change itself was seen only through instruction/SLM counts (1290->1202 instCount), **no standalone time measurement** | `sdpa-ocl-decode-kernel` |
| `slm_p` layout | must be key-indexed with head as the innermost vector element. Written head-major it becomes 64 separate 32 B SLM reads (SLM load 44->72, instCount +12%) | same |
| ISA count metric | the first version once went 1290->1440, **slower** (slm_p layout, 18 scalar loads for the page lookup) | same |
| Excluding SLM | when 4 or more subgroups split the key axis, `slm_out` + a barrier are added: with `V_TILES < SG_PER_WG` (head < 128) sg16 is neutral to a loss, head 256/512 (V_TILES 16/32) is free | `sdpa-ocl-decode-tiling-sg-per-wg` (MEASURED B70) |
| SLM-shared dequant | share the 4x redundant u4 V dequant through SLM: +16 KB, WG 4->3, dropped | §5.2 |

SLM transpose (DG2 S2 K3) was the worst of the K-direction candidates (D=128 t_med: K0 contiguous 286 us, K1 scalar gather 365 us adopted, K2 vload8+pack 370, **K3 SLM transpose 449**, K4 A=Q/B=K 344) (MEASURED, DG2, `sdpa-ocl-xe-hpg-facts`).

## 7. Software prefetch

### 7.1 Where it is used (current code)

| Location | What | Condition |
|---|---|---|
| decode V | `PREFETCH_V_TILE`: `intel_sub_group_2d_block_prefetch_16b_16r16x1c` / `_8b_32r16x1c` for i8·u4 (`sdpa_ocl_decode.cl:177-197`), `PREFETCH_DIST` chunks ahead (default 4, `SDPA_OCL_DECODE_PREFETCH`, `sdpa_gen_ocl_decode.cpp:256`) | `USE_PREFETCH_V && USE_2D_BLOCK_IO_V` |
| MIXED Vc | `vc_prefetch` (`sdpa_ocl_v_load.cl:21-40`): **right after** the S_max aggregation barrier, before the softmax qb loop. Issued only when `!from_cache && sg_i_sv == 0` | `IS_PA_K_U4 && PA_CUR_KV_F16` (u4 only) |
| PA PREFILL V, `d_max` 512 | `v_tile_prefetch_l3` (`sdpa_ocl_v_load.cl`): address prefetch of this key tile's V rows into L3 only, one address per 256 B line (§7.6) | `PREFETCH_V_L3` (`v_prefetch_l3`, `SDPA_OCL_V_PREFETCH_L3`); replaces `PREFETCH_V` |
| K | **none** (decode K prefetch was deleted) | |
| mask | none (micro has `PREFETCH_MASK`) | |

Note: a 2D block prefetch needs no destination register, so it gives "memory-level parallelism (MLP) that occupancy cannot buy" (comment at `sdpa_ocl_decode.cl:177-182`). Each prefetch costs about 8 instructions of a64 address/descriptor setup (32 of them took instCount 1202 -> 1469, mostly mov).
8b prefetch builtins exist (§1), so 2D prefetch is kept for i8/u4 too. The call site must use **the same surface/base fixup/coordinates** as the actual read (`Vc_b2d`, `VcD_w_b2d`, `VcD_x0` in `vc_prefetch`).

### 7.2 Measured results (decode, llama-3.1-8b, head 128, M=4, `sdpa-ocl-decode-kernel` 2026-08-10; HW not recorded in the note, B70-migration era)

| Experiment | Result | Interpretation |
|---|---|---|
| K prefetch (`prefetch_32b_16r8x1c`) | **3.6% slower** | the KQ loop already interleaves `KEY_GROUPS` independent DPAS chains over a single 4 KB page -> the loads were already pipelined |
| V prefetch | **2.1% faster** (default ON) | S*V walks 16 different pages with a single accumulator chain, so the loads were exposed |
| both K+V | +0.8% | the K loss eats the V gain |
| distance | 1: -1.45%, 2: -1.87%, **4: -2.10%**, 8: -0.56%, 16: **+2.21%** (non-monotonic) | distance changes **only where they are issued**, not how many: `min(dist, chunks)` go in the barrier window, the rest one iteration ahead. With distance > ~4 the in-flight set (8 KB/subgroup x 8) overflows L1 and is evicted before use |
| "push everything into the barrier window" | **wrong** | the non-monotonic result above |
| absolute time | prefetch off 1.1194e9 ns vs distance 1/2/4: 1.1031e9 / 1.0984e9 / 1.0958e9 | `docs/sdpa_ocl.md` "V is prefetched ..." |
| share taken by the split arrive/wait in the barrier window | 0.7%p of 2.1%p | §8 |

### 7.3 MIXED Vc prefetch (u4, llama-3.2-1b-instruct, B70, cliloader average ns)

V_PREFETCH on 145,603 ns vs off 151,866 ns = **-4.1240%** (64 calls, `sdpa_ocl_mixed_..._sa`, SIMD16 REG128 SLM=17664, GWS[272x512x1], LWS[16x16x1]); micro 147,825 ns;
**-8.23%** in total vs the first exact baseline 158,661 ns (`sdpa-ocl-mixed-exact-complete`, MEASURED). **Never measured on i8 BY_CHANNEL MIXED** (u4-only `#if`).

### 7.5 PA PREFILL V prefetch (2026-10-23, B70 pinned 2800 MHz, per-call device time, `sdpa-ocl-longctx-prefill`)

`v_tile_prefetch()` (`sdpa_ocl_v_load.cl`): after the S_max atomics, the workgroup cooperatively issues
`intel_sub_group_2d_block_prefetch_16b_16r16x2c` over this key tile's V rows (`(wgTK/16) * ceil(V_HEAD_SIZE*2/64)` blocks, stepping by
`sg_per_wg`), inside the split barrier window (§8), where `sdpa_micro` prefetches V. `PREFETCH_V` (`SDPA_OCL_V_PREFETCH`), PA PREFILL with
2D K/V reads only; `#error` for MIXED / no 2D KV / SG8.

| Shape | off | on | Note |
|---|---|---|---|
| qwen3-8b head 128, 8K, 256 x 128 tile (with branchless exp) | 7,914 us | 7,555 us (**-4.5%**) | then the split barrier: 7,218 |
| gemma-4-e2b head 512, 8K, 128 x 64 tile (256GRF, alone) | 8,880 us | 9,762 us (**+10%**) | a 128-key tile of 512-channel V rows is 128 KiB |

-> `v_prefetch` is a per-tiling flag, set at `d_max` <= 256 only. K prefetch in the same place: next tile +0.2%, first tile +0.4%,
both + V +3.6% (rejected). The "tile byte size" reading was wrong: the cost is the line requests, 4 per 256 B line with a 64 B
wide block prefetch on 1 KiB rows (§7.6, MEASURED 2026-10-24).

### 7.6 PA PREFILL V L3 prefetch at `d_max` 512 (2026-10-24, B70, `sdpa-ocl-head512-prefill-task`)

Problem: the head 512 PA PREFILL kernel had a knee at 16K-20K keys at a **fixed** 1800 MHz clock (time / keys^2 flat to 16K, +9.6% by
32K); `sdpa_micro` had none. One KV head's K+V is 2 KiB per token, 24 MiB L3 = 12K tokens. VTune: L3 miss 3.0% -> 18.2%, Send
stall 20% -> 30%, DRAM read 64 GB/s (latency, not bandwidth). `PREFETCH_V` removed the knee (flat 121 ps) but cost +14% at 8K.

Cost model, measured with an address prefetch (`__builtin_IB_lsc_prefetch_global_uint`, one address per lane, this tile's V,
gemma-4-e2b, per call against no prefetch, 1800 MHz, two rounds within 0.3%):

| Stride, cache control | 8K | 32K |
|---|---|---|
| 64 B, `L1UC_L3C` (4 requests per 256 B line) | +17.0% | +4.5% |
| 64 B, `L1C_L3C` | +15.0% | +4.4% |
| 128 B, `L1UC_L3C` | +4.4% | -5.8% |
| **256 B, `L1UC_L3C` (one request per line)** | **-0.5%** | **-11.0%** |
| 256 B, `L1C_L3C` | +1.2% | -8.9% |
| 512 B, `L1UC_L3C` (every other line) | -0.1% | -3.5% |

- **The cost of a prefetch is the number of line requests it adds to L3**, not the message count (32-row x 2-block K/V *demand* reads
  with 4x fewer messages for the same lines were 4-14% slower, one run) and not the cache level at the 64 B stride (`L1C_L3C` ~
  `L1UC_L3C`). At one request per line, filling L1 too still cost 1.6 / 2.1 points. 512 B kept about a third of the gain: L3
  fills whole 256 B lines (`clinfo` "Global Memory cache line size", B70), so the step must equal the line, not exceed it.
- Product (`39bdb18aeb`): `v_tile_prefetch_l3()` at the `PREFETCH_V` site (after the S_max atomics, inside the split barrier),
  `__builtin_IB_lsc_prefetch_global_uchar(V + row * pitch + line * 256, 0, 2u /* L1UC_L3C */)`, `kq_wg_tile_keys x ceil(row/256)`
  lines dealt over all lanes (head 512: 1,024 lines / 512 lanes), rows clamped to `k - 1` (no address leaves the tensor; the byte
  form needs no alignment). Result per call against no prefetch, 1800 MHz: -1.6 / -0.8 / -0.4 / -1.5 / -2.4 / -5.6 / -7.9 / -9.4 /
  -11.1% at 2K..32K; flat 105-106 ps from 12K. At the 2800 pin (PL2 active) 32K 111.8 -> 100.1 ms; TTFT -1.3..-1.9% at 32K.
- Other placements, 256 B stride: the next tile's V -10.1% at 32K, this tile's V + the next tile's K -8.6% (+4.0% at 8K), the
  next tile's K + V -6.8%. K prefetch does not pay here either.
- **Where it does not pay**: in place of `PREFETCH_V` at `d_max` <= 256 (one KV head's K/V fits L3 at 32K): head 128 +3.8 / +3.9%,
  head 64 +5.1 / +5.2%, head 256 window layers +2.6..+2.9%, head 256 full layers -0.8 / -1.0% (8K / 32K). Prefetch where the data
  is already in L3 only adds requests.
- **Declaring the builtin in a kernel file**: `tile_ops.cl` declares `__builtin_IB_lsc_prefetch_global_*` with `enum LSC_LDCC`. The
  OCL codegen (`primitive_db_gen.py` `append_undefs`) appends an `#undef` for every `#define` of a kernel file after each kernel,
  and a batch puts several kernels of one file into one program, so a guarded `enum` there fails with "redefinition of
  'LSC_LDCC'" in the second kernel. Declare the cache control as `uint` (the enum is compatible with `unsigned int`; ocloc compiles
  both prototypes in one translation unit) and guard with `!defined(GPU_OCL_TILE_OPS_H)`.
- A prefetch changes no value: 173 PA gtest outputs bit-identical with the switch forced on/off at every head size.

### 7.7 i8 MIXED page prefetch (2026-10-27, B70 1800 MHz, cliloader per call, 8K, 256-token chunks)

`PA_MIXED_PREFETCH_V` (this key tile's V block columns at iteration start, the u4 helper) turned on for i8 too, on **every**
tiling including the `d_max` 64 table one: on top of the exact widen, qwen3.5-9b 799 -> 709 us, llama-3.1-8b 1,167 -> 972,
gemma-3-4b 563 -> 490, gemma-4-e2b `d_max` 512 955 -> 902, llama-3.2-1b (table) 630 -> 527, gpt-oss-20b 1,171 -> 961.
`PA_MIXED_PREFETCH_K` for i8: 0 to +1.6% on the same models (kept off). u4 at `d_max` 64 reads V with the 1D page read, so the V
prefetch (block2d only) does not apply there.

### 7.4 Failed/invalid prefetch

- In plain prefill, issuing the V prefetch inside the k0 iteration right before consuming the same tile: **about 11 us net loss** (right before consumption there is no compute to hide it behind; pure overhead). Code removed (the `#if 0` block just before `ddd9ab8ef4`, `sdpa_ocl.cl` history).
- `MAX_BARRIER_V_PREFETCH` (V prefetch inside the split barrier window of plain prefill) came in as a toggle in `ddd9ab8ef4` and was used as `SDPA_OCL_256GRF=1 ... MAX_BARRIER_V_PREFETCH=1` in the llama-2-7b 4096-token bench (`test/run.sh`), but
  **its gain was never proven** and it was deleted in the 2026-09 refactor (`sdpa-ocl-refactor-2026-09`: "never shown to help"). No individual measurement result file was found -> the effect is recorded only as "unproven".
- micro uses `PREFETCH_K/V/MASK` one k0 iteration ahead into L1+L3 (`LSC_LDCC_L1C_L3C`, `cooperative_prefetch_2d_*`). Its shape: one
  `__builtin_IB_lsc_prefetch_global_uint/uchar` per lane, one 64 B address each, and only the first `PREFETCH_D_MAX = min(d_max, 64)`
  elements of each row (128 B of a 1 KiB head 512 row), i.e. one prefetch message per subgroup per tile (`sdpa_gen_micro.cpp`,
  `tile_ops.cl`). sdpa_ocl took the address form, not the coverage, in §7.6.
- DG2: generic `prefetch()` has no effect (MEASURED, S2 P8: recorded only).

## 8. Split barrier (`intel_work_group_barrier_arrive/wait`)

Actual code locations (current working tree):

| Location | Code | Intent |
|---|---|---|
| `sdpa_ocl.cl:766` | `barrier(CLK_LOCAL_MEM_FENCE)` | `S_max_slm` atomic max aggregation complete. Followed by `vc_prefetch` (`:770`) |
| `sdpa_ocl.cl:842` `intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE)` ... `:868` `intel_work_group_barrier_wait(...)` | arrive after the softmax `S_slm` write/`S_sum_slm`, **rescale A_tile by `alpha` in between** (when `!first`), then wait | overlaps the sync that lets pA be read only after every subgroup has written S_slm with the "A_tile rescale" compute |
| `sdpa_ocl_decode.cl:727-732` | arrive -> `PREFETCH_V_TILE` x `min(PREFETCH_DIST, CHUNKS)` -> wait | fills the barrier wait after the slm_p/slm_sum writes with prefetches of the first V tiles (comment at `:721-722`) |
| `sdpa_ocl_decode.cl:734` | `#else barrier(...)` (prefetch off) | |
| `sdpa_ocl.cl` PA PREFILL, `MAX_SPLIT_BARRIER` (2026-10-23) | the S_max barrier becomes arrive -> `v_tile_prefetch` (§7.5) -> wait | as `sdpa_micro`: the V prefetch is issued while the other subgroups finish their atomics |

Rules (based on code comments and notes):
- Between arrive and wait, place only independent work that **does not read the SLM data that barrier protects** (alpha rescale works on the private A_tile, prefetch is unrelated to SLM).
- Inside a split barrier, **do not change the number or placement of the barriers themselves**. MIXED's `SV_TRIM` skips blocks with `cp*DPAS_K >= k_chunk` via `continue` but keeps the barriers outside the loop (`sdpa_ocl.cl:891`, "the barriers stay outside").
- The local fmax atomic (`__builtin_IB_atomic_max_local_f32`) and the split barrier also PASS on DG2 SG8 real hardware (S2 mini SDPA 36/36). `cl_intel_split_work_group_barrier`.
- Measured effect: of the 2.1% decode V prefetch gain, **0.7%p** comes from issuing in the barrier window (MEASURED, `sdpa-ocl-decode-kernel`). No record that isolates the **standalone** effect of the plain prefill split barrier was found -> left as ASSUMED. PA PREFILL (MEASURED 2026-10-23, B70 pinned): qwen3-8b 8K 7,555 -> 7,218 us (-4.5%) on top of the V prefetch, +5.4% when switched off again; at `d_max` 512 with nothing in the window (no V prefetch) 0%. The 4.12% of MIXED V_PREFETCH is issued right after a barrier and does not change the barrier itself ("no new barrier was created").
- Other history: the tk=32 correctness bug investigation put compiler ordering differences around the split barrier on the suspect list, but this was not proven (`sdpa-ocl-tk32-bug-hunt`, UNRESOLVED).

## 9. Scalar gather and the K cache layout (d-major -> token-major)

### 9.1 Why K was a SIMD-1 scalar gather

The PA K cache (original) is `[num_blocks, kv_heads, k_head_size, block_size]` (**token innermost, d-major**), V is `[.., block_size, v_head_size]` (**d innermost, token-major**)
(`paged_attention_gpu_test.h`, `paged_attention_opt.cl:105-106`, `plugin/ops/paged_attention.cpp:45-46`; verified 2026-08-05). The KQ A operand wants lane=head(d), element=key.
- For V (token-major), `transform_16b_16r16x1c` applies directly to the page.
- For K (d-major), a coalesced read gives lane=token (the wrong axis) -> **per-key SIMD-1 scalar gather**: head 128 mixed ocloc total ops 2651, 128 K scalar gathers.
- Every way out is blocked: transpose is 32b only (packing two f16 puts two *tokens* in one dword), the pitch is 32 B (< 64 B), and the operand swap (A=Q, B=K) is what micro does, but it yields S^T, turning the per-query softmax max/sum from free in-register work into a cross-lane `sub_group_reduce`.
- **Fix**: store K token-major like V, `[num_blocks, kv_heads, block_size, k_head_size]`. Then the K page has the same geometry as prefill -> call `read_16b_16r16x1c` with `(x=db*DPAS_K, y=0)` and pitch `HEAD_SIZE*2` (`sdpa_ocl_qk_load.cl:330`).
  (ISA, head 128 mixed) total ops 2651 -> 1353 (-49%), K scalar gathers 128 -> **0**, dpas 32 unchanged. head 64 mixed instCount 2322 -> 1288 (-45%), `load.ugm.d16u32` 64 -> 0.

### 9.2 relayout ladder (2026-08-05..06, `pa-k-token-major-progress`)

| rung | Content | Verification |
|---|---|---|
| 1 | dim order (`keyCacheDimOrder` `transformations_pipeline.cpp`), writer (`pa_kv_cache_update_ref.cl` KEY_TOKEN_STRIDE/KEY_HIDDEN_STRIDE), decode reader | toggle off 129/129, on 124/129 (5 expected MIXED failures) |
| 2 | sdpa_ocl MIXED K via `read_16b_16r16x1c` on the page (`USE_2D_BLOCK_IO_K_PA`) | all 3 paths 129/129, the actual `sdpa_ocl__generate` confirmed with cliloader |
| 3 | rotate | **real bug found**: the d-major kernel paired *different tokens* on the token-major cache, corrupting all 512 pairs. All 59 cases of the default suite passed (blind). After adding `smoke_kv_cache_rotation_content` (4), which reads the cache back, 4/4 FAIL -> PASS |
| 4 | adaptive_rkv, reorder | adaptive_rkv 15 FAIL -> 24/24; `pa_kv_reorder_gpu` was **blind because the test shared the same d-major assumption** (8/8 passed); kernel and test fixed together |

Lesson: **a green suite is not evidence that a consumer matches the layout.** Check that there is a test that actually observes the values that kernel writes.
`IS_KEY_TOKEN_MAJOR` touches 5 kernels (decode, writer, rotate, adaptive_rkv, reorder). Single origin: `keyCacheDimOrder`; XAttention/CM are already token-major.

`pa_sdpa_opt` (decode) on token-major reads "each lane its own token row" with `vload16(0, key_cache + block_offset + sglid*K_HEAD_SIZE + ...)`. Correct, but **not faster** (the user's measurement: no change to slightly slower).
Why (ISA): d-major's `load.ugm.d32x8t (1|M0)` was a **block load** (SIMD-1: one address distributes 32 B of contiguous data across the lanes, already optimal), which was **misread** as "demoted to a scalar gather" (twice in one session).
`vload16` is a **scatter** of 16 `load.ugm.d32 (16|M0)` to different addresses (256 B apart per lane): per qk_idx, d-major 16 block messages (~8 cache lines) vs token-major 8 scatters (16 lines) -> fewer instructions but more traffic.
**Metric lesson: when the access pattern changes, instCount/message count is the wrong metric** (improved 1076->980, loads 178->114, but wall-clock unchanged). Compare the message *kind* (block vs scatter vs transform) and the number of touched lines.
On top of that, `transpose_32b_16r8x1c` could cut the 8 scatters to 1 block, but **this was blocked because `pa_sdpa_opt` must also run on pre-Xe2** (the user's decision 2026-08-06): it needs an arch gate + a non-2D fallback.

2026-10-27: sdpa_ocl MIXED now reads the **d-major** f16 page with block reads (§4.5) at token-major speed (406 vs 401 us per
call on qwen3.5-9b), so for f16 MIXED the relayout no longer buys speed; `OV_GPU_PA_K_TOKEN_MAJOR` stays a staging switch.

### 9.3 Page layout table (current `docs/sdpa_ocl.md` "Paged-attention cache layouts", h = head size, block 16)

| Cache | data row pitch | comp area | K page stride |
|---|---|---|---|
| f16 | `2h` | none | `32h` |
| i8 BY_TOKEN | `h` (**not `h+4`**: the +4 is room for the per-token f16 scale/zp arrays at the end of the page) | per-token scale `[t]`, zp `[16+t]` | `16*(h+4)` |
| i8 BY_CHANNEL (token-major) | `h` | per-channel (scale, zp) f16 pair, one dword `[c]` | `20h` |
| u4 BY_CHANNEL K (token-major) | `h/2` (**not aligned**: `16*(h/2)+4h = 12h` matches the upstream d-major INT4 page allocation byte for byte) | same as i8 BY_CHANNEL | `12h` |
| u4 BY_TOKEN V | `Align(h/2, 16)` (absorbed by the trailing comp slack: `16*PV+64 == 16*(PV+4)`) | same as i8 BY_TOKEN | `16*(Align(h/2,16)+4)` |

- The page size is **unchanged** by the d-major <-> token-major switch (`k_head_size*(block_size+4)`) -> allocation/pool sizes unchanged, only addresses inside the page move.
- **For K, BY_TOKEN keeps its comp on the token axis even when token-major, while BY_CHANNEL in token-major has its comp on the channel axis** (scale/zp depend on d -> matches the KQ depth axis folded into Q; the A operand differs per page, so the `AS_A(qv)` rebuild moves inside the g loop, a structural cost: +Q_PER_WG half multiplies/tile).
- BY_CHANNEL comp is `4*k_head_size` = 512 B/page (head 128), larger than the 64 B of BY_TOKEN: K page 2560 B vs 2112 B (+21%) - inherent to the format (K+V ctx 4352, 8 kv-heads: 10.2 MB vs 9.2 MB, f16 17.8 MB). Kernels cannot reduce it.
- There are only 4 consumers (writer, rotate, sdpa_ocl MIXED, sdpa_ocl_decode); the rest assume d-major. The layout is decided once per model in `transformations_pipeline.cpp`, and consumers **derive it from the physical shape of the cache**
  (`k_by_channel_token_major_layout(shape, adjusted_block)`; token-major if dim 2 is the adjusted block). For one mismatch direction, `PagedAttentionOptImpl::update_rt_params()` throws and `pa_kv_reorder` rejects
  (`pa-mixed-layout-gate-use-ocl-hole`: the hole where the MIXED K-page check was gated on `use_ocl`, so with `TEST_USE_SDPA_OCL=0` micro-MIXED read a token-major page -> `96b457a4f0`).
- 3 facts (do not re-derive, `sdpa-ocl-pa-i8-by-token-task`): (1) `OV_GPU_PA_K_TOKEN_MAJOR=1` being a no-op for i8 caches was a scope limit of *our code* (`aaf3ca3ae2`). (2) The row pitch of i8 BY_TOKEN is `h`.
  Mistaking it for `h+4` led to the wrong conclusion "132 B pitch -> 2D block impossible", which killed a design option. (3) PREFILL does not read the K/V cache; it reads the f16 input tensors (`sdpa_gen_ocl.cpp:594`).

### 9.4 per-key scale/zp loads were gathers too

In plain i8 prefill, the 128 SIMD-1 `(1|M0)` loads turned out to be not K data but **per-key scale/zp** (per-key values read in the innermost position are lane-uniform, so IGC issues one message per lane, with 15/16 lanes idle).
Fix: one subgroup-cooperative wide load (lane=key) at the top of the k0 tile + `sub_group_broadcast` (the lane is a compile-time constant, so it folds into a region). In PA MIXED i8 too, 256 -> 0 per k0 (head 128, instCount 5450 -> 4868, ISA).
Switching to `block_read` was rejected because it is already coalesced into 1 SIMD16 gather message, so the gain ≈ 0 (ISA).
The `block_indices[]` page lookups are hoisted the same way: `DKS*kq_key_blocks*DPAS_ROWS` times (head 64: 64) -> `kq_key_blocks` times. In decode one partition covers exactly `SUBGROUP_SIZE` pages, so the page table is read once per lane, then `sub_group_broadcast` (global loads 53 -> 36, avoiding 18 scalar loads).

## 10. Write path: the kvup writer and the store sector floor

### 10.1 u4 token-major writer regression and fix (`pa-kvup-u4-token-major-writer`, 2026-08-13)

Token-major breaks the contiguity of the "token vector of a channel" that the requantize loop walks. In d-major 16 nibbles are 8 contiguous bytes, so IGC merges the read and the write into one wide message (2 tokens/trip). In token-major they are `K_HEAD_SIZE/2` apart and do not merge, and
since a byte is a channel pair, channel-per-lane needed `intel_sub_group_shuffle` + a divergent `if (!hi_nibble)` for every token. Instruction slope per token: d-major 18.3 vs token-major 40.3.
Fix: a lane owns **a byte and a token parity** (`col = sglid % (SG/2)`, `par = sglid / (SG/2)`, `pa_kv_cache_update_ref.cl:391-392`), min/max via 1 butterfly (`sglid ^ half_sg`) (fmax/fmin are commutative/associative, so **bit-identical**).
**Three traps (each measured)**: (1) a loop variable that varies by lane steals the array's registers (`for (t = par; ...; t += 2)` took `r[a0]` 2 -> 274, **1.8x slower**) -> use uniform `it/npo/np`. (2) The **shape** of a uniform guard decides whether the inner loop unrolls:
loop bound set to 0: unroll lost (7850 -> 2590 static); `if (skip) return;` at the top of an inline helper: unroll lost + **18.7 us (2.3x slower than the unsplit 8.2 us)**; the same check at the **call site**: unroll kept + 5.6 us. Put the guard at the call site, not in the helper body or the loop bound.
(3) Preserve type promotion per expression (`(max==min) ? 0.004 : (max-min)` subtracts in half, then widens).
V in a separate WG: V is BY_TOKEN, so one WG must own the reduce over the whole head dim. A gain **only in prefill** (the prefill WG also does all K channels, so max(K,V)): in the ablation, generate floor 580, V-only 895, K-only 2755, K+V 2777 ns -> V costs 315 ns alone but 22 ns in situ (absorbed by idle issue slots of K's long dependent chain).
`HAS_SEPARATE_V_WG_PREFILL` touches only prefill, and `get_global_id(2)` -> `get_local_id(2)` is **mandatory** (because gws[2] spans both WGs).
Measured (probe B70 head 128, 8 kv heads, ns/dispatch): generate u4 d-major 2904 / token-major (regressed) 3427 / **fixed 2776**, prefill (2 blk) 29444 / 29920 / **14324**.
**real-model (llama-3.1-8b, cliloader, debug build)**: generate `[1x8x128]` 31620 calls 192,259,083 (d-major) / 216,920,855 (regressed) / **192,274,023 (fixed)** -> **regression fully removed, exactly tied with d-major** (the -4.4% the probe predicted did not materialize: the real run carries ~3200 ns/dispatch more that the probe lacks, which dilutes it).
prefill 1984 calls 74,837,410 -> 76,483,353 -> **36,547,697** (-52.2%, matches the probe's -52.1%; `GWS[16x8x32]` is evidence of the extra V WG dispatch). Totals 275,999,683 / 303,455,125 / **237,128,428** (-21.9% vs regressed, -14.1% vs upstream d-major).
The probe **predicts large ratios well but not small ones** (it under-reports absolute time by about 2.2x on generate).

### 10.2 store-sector floor (`pa-kvup-token-major-store-sector-floor`, 2026-08-18, B70)

Remaining regression: the head-256/kv-8 generate dispatch (`GWS[1x8x256] LWS[1x1x16]`) went 6312 -> 7074 ns (**+12.1%**); head-512 is unchanged (8099 -> 7956). Cause (ablation): one thread owns **8 bytes x 16 token rows** and the row spacing is `BC_TOKEN_STRIDE = K/2 = 128 B`, so
it **dirties 16 different 128 B sectors**. In d-major the 16 channels are `16*ADJUSTED_BLOCK = 192` contiguous, line-aligned, exclusively owned bytes = **2 sectors**.

| ablation (ns/dispatch, head 256) | floor | old-token loads | **data stores** | total |
|---|---|---|---|---|
| d-major | 2201 | 723 | **423** | 3353 |
| token-major | **1283** | 889 | **2014** | 3700 |

The store cost is proportional to the number of sector writes in the whole dispatch (~0.5-0.8 ns/sector): K=128 1024 sectors -> 912 ns, K=256 2048 -> 2014, K=512 (2 groups, so per-thread reuse) 512 -> 746.
Ceiling (timing-only probe, head 256): sectors/thread 16 -> 3698, 8 -> 2851, ~1 -> 2712 -> **total opportunity 986 ns** (0.024 ms/token), of which 16 -> 8 accounts for 847.
**ISA/message count was the wrong metric**: token-major had fewer instructions (4249 vs 4448) and fewer indirect `r[a0]` (125 vs 188), with the same number of store messages (50 vs 51), yet the stores were 4.8x slower. Only the ablation found it.
The four candidate fixes were **all rejected by measurement** (§12 table). Conclusion: no code change, accept +0.019 ms/token, spend the effort on the remaining 89% of the e2e gap.
(That e2e gap itself: wall clock 23.3 s vs device 16.6 s, i.e. 29% of the wall time is outside the device, 1,680,102 enqueues. Unverified hypothesis: a fixed cost per enqueue.)

Other: `reqd_work_group_size(1,1,SUBGROUP_SIZE)` (`pa_kv_cache_update_ref.cl:807`) rejects `lws[2] != 16` with CL_INVALID_WORK_GROUP_SIZE (-54) -> widening the generate lws requires removing the attribute (prefill keeps 16).
i8 BY_CHANNEL token-major is slower than d-major **at every h_sub** (parts=16 2882 vs 2432, +18.5%; prefill +10%): a different cause than u4 - the non-int4 arm turns the d-major one-column `vload16` into 16 scalar gathers (a message count regression, not sectors). gemma-4 (u4) is unaffected. Not fixed.

### 10.3 IGC 2.41: the writer's private arrays went to scratch, and the store order (`pa-kvup-writer-igc241-spill-task`, 2026-10-27, B70 1800 MHz, `2721e30d44`)

After the 2026-10-26 driver upgrade (IGC 2.36.3 -> 2.41.5) the u4 token-major writer reported `TPM=576` at every head size and its PREFILL dispatch became 1.3-1.6x master's d-major writer (qwen3-8b 8K 322 vs 224 us).
**Not register spill**: the `.asm` header had `//.private memory size 576` and no `//.spill size`; `*_optimized.ll` kept one `alloca [2 x [9 x half]]` (`token_vals` of the requantize helper, into which LLVM had merged the PREFILL helper's `input_data[2][9]`). 576 = 2 x 9 halves x 16 lanes. Every index was subgroup-uniform but runtime (`for (it < npo)`, `[npo]`, `[t / 2]`, `[np]`): IGC 2.36 promoted that, 2.41 does not.
256 GRF, `IGC_ByPassAllocaSizeHeuristic=1`, `IGC_DisablePromotePrivMem=0` and `IGC_LLVMCommandLine=-disable-inlined-alloca-merging` all left 576.
Fix: compile-time indices only. Loops run over all 8 pairs as `unroll_for` with a uniform `if (it < n)`; a value whose slot is only known at run time (leftover old token, a new token, the requantize odd tail) is written or read through a select over every slot; the PREFILL odd tail gets its own variable.

Second-order effect: with the scratch gone, head-256 GENERATE was **slower** than before (probe 5.75 vs 5.35 us). TIMING-ONLY ablation: removing the 8 data stores gave 3.99 (old code 3.98) -> the store tail. The 16 partitions of a kv head each write 8 B of the **same** 128 B token rows; issued in the same row order they queue on the lines, and the old code's scratch reloads had spaced them out.
Fix: quantize all 8 pairs into a `uchar8`, store from pair `get_group_id(2) % 8` (`qb[itr]`, one indirect register read per store), only where `NUM_HEAD_SIZE_GROUPS == NUM_K_HEAD_SIZE_PARTITIONS && BC_TOKEN_STRIDE >= 128` (head 256): 5.03 us. At head 128 (+3-8%) and 512 (+5%) the rotation costs more than it saves. This is the first lever that moved the §10.2 floor: ordering across threads matters even though co-location (§12 #16) did not.

| us per call, master / before / after | PREFILL 8K | PREFILL 32K | GENERATE |
|---|---|---|---|
| qwen3-8b (h128, 8 kv) | 224 / 322 / 172 | 778 / 1219 / 570 | 8.3 / 8.9 / 8.1 |
| llama-3.2-1b (h64, 8 kv) | 121 / 156 / 82 | 437 / 621 / 318 | 7.7 / 8.4 / 7.6 |
| gemma-4-26b h256 (8 kv) | 400 / 592 / 274 | 1483 / 2392 / 1032 | 9.2 / 10.1 / 9.5 |
| gemma-4-26b h512 (2 kv) | 270 / 354 / 133 | 932 / 1279 / 574 | 11.9 / 11.9 / 11.9 |

Cache bytes bit-identical (186 basic rows: output, K and V cache dumps). The basic gtest itself passed two writer variants that corrupted the K cache of 13 / 10 rows: compare dumped cache bytes, not the attention output (chapter 05 H).
Tools: `test/kvB` (`splice.sh` for an IGC batch dump with several kernels, `isa.sh`, `probe` with `--bench-blocks/--bench-past`, `heads.sh`, `sweep.sh`, `modes_ab.sh`, `patch/pa_dump_kv_hook.diff`).

## 11. Dynamic shape, padding, OOB risks

| Risk | Fact | Mitigation |
|---|---|---|
| 2D block OOB | HW fills with 0 (outside width/height). **A global scalar OOB read on Xe2 gives not 0 but CL_OUT_OF_RESOURCES or garbage** | clamp the 2D block surface to the valid region; always guard scalar reads |
| unwritten page slots | NaN can survive into the scores even after masking (`-inf + NaN`) | height clamp (`kp_rows`); for V force scale AND zp to 0 (comp comment in `sdpa_ocl_decode.cl`; unlike K, V has no place where it is overwritten later) |
| page lookup for key >= k | page 0 is always allocated -> out-of-range `block_indices` read page 0 (`pa_v_page_base` `sdpa_ocl_v_load.cl:11`) | the probability of these keys is exactly 0, so the values only need to be finite (the reason the `1D` path removed the per-key guard) |
| last k0 tile | `key_base` can exceed `k` -> illegal read with height <= 0 | zero-fill (the `else` branch of `pa_k_tile_b2d16`) |
| head tail (`d % DPAS_K != 0`) | with surface width `d*sizeof(T)` the HW zero-fills; no leak into the next head | `KP_w = d * sizeof(half)` |
| dynamic padding | sizes arrive at runtime via shape_info | the fixup tier trusts the assumptions, the strict tier is false (§3.4); wrong result if violated (residual risk) |
| dynamic mask shape | if the trailing dim is dynamic at JIT time, it is inferred as `mask_kind = m_is_prefill ? 2 : 1`. If the runtime shape is `[B,H,1,K]` (per-key), the kind 2 assumption reads a 1-row buffer for q rows (up to 15x): **CL_OUT_OF_RESOURCES/NaN** (`sdpa_gpu_test_64_32_990_128_2`, 2026-09-17, `test/SDPA_OCL_MASK_KIND_OOB_SESSION.md`) | in the kernel, `MSK_D2 == 1 ? 0 : row`, broadcast if `MSK_D3 == 1`, upper bound `MSK_D2/D3` (static masks fold at compile time into byte-identical codegen; only dynamic adds 1 compare to the prologue). Reverting to `MASK_KIND=-1` was not adopted because of an IGC regression. micro is already safe via the runtime `MSK_D2 == 1 && MSK_D3 > 1` check at `sdpa_micro.cl:620` |
| u4 Kc dword alignment | depending on runtime padding, the half offset of the first channel of a head can be odd | `kc_dword_ok` runtime check + exact scalar Kc fallback (falling back to the cache is wrong: the current token in raw and in the cache differ numerically) |
| PA scores output/adaptive RKV/alibi/qq_bias | read the token-major BY_CHANNEL page as d-major -> garbage | layout gate `allow = !has_scores_output && !has_adaptive_rkv` (`PA_BY_CHANNEL_TOKEN_MAJOR_SCORES_DECODE_SESSION.md`, 2026-09-21): the root cause of the `/6` cases failing even with `TEST_USE_SDPA_OCL=0` was that the writer wrote token-major while `paged_attention_opt__single_token` did not jit `IS_KEY_BY_CHANNEL_TOKEN_MAJOR` and read it as d-major |
| i8 BY_CHANNEL requantize | `pa_kv_cache_update_ref.cl:297` ignores `in_data_pitch` -> misreads padded keys at new tokens j>=1 | **not fixed**, recorded in Known issues (`sdpa-ocl-ki-t7`) |

### 11.1 V page split bug (`test/V_PAGE_SPLIT_BUG_HANDOFF.md`, historical)

When MIXED reads `[0,past_len)` from the cache and `[past_len,k)` from raw Kc/Vc, putting the source switch at `align_up(past_len, PAGED_ATTENTION_BLOCK_SIZE)` (old GRAN=1 version, page granularity) made **K correct and only V wrong**.
Symptom rule: when `pa_key_end` is **not** a multiple of `kq_wg_tile_keys`, one k0 iteration mixes both sources. K's `from_cache = (key_base < pa_key_end)` follows the KQ key split (per subgroup), while V's `v_from_cache = ((k0 + cp*SUBGROUP_SIZE) < pa_key_end)` follows the S*V `cp` split: **different mappings**.
It also fails with an f16 cache (cache read and Vc read are bit-identical), so it is not quantization error. The **mechanism of this bug was never identified** (many suspects ruled out: if/else nesting, surface descriptor, Vc pointer, base fixup, y>=0, parity/VNNI row pairing, V page block read).
**The fix was a design change, not a root cause**: the meaning of `GRAN=1` was changed to "exact split that cuts exactly at past_len". `k_chunk` is clipped at `past_len`, the chunk tail mask/`last` are computed from the actual chunk, and the loop advances with `k0 += k_chunk` -> the source choice of every iteration is **WG-uniform** (same as micro PR #37377).
(2026-09-09, `sdpa-ocl-mixed-exact-complete`; the GRAN/SIDE/page-rounded paths were deleted in the 2026-09 refactor.) Lesson: **a design in which the source differs per subgroup within one k0 tile is dangerous because K and V use different key splits - create a WG-uniform boundary.**
The "GRAN=0 is correct" claim and the failure counts in old documents are not current evidence.

## 12. Failed/rejected experiments table

| # | Attempt | Result | Cause (or marked hypothesis) | Source / HW |
|---|---|---|---|---|
| 1 | int8 K via `8b_16r16x4c` (head 64, assuming no shuffle) + scale folding | variant1 5888 / 30,780 ns **slower than scalar** (correctness 1024/1024 PASS) | shuffle/extra work cancels the coalescing gain. spill is not the bottleneck | `sdpa-ocl-int8-perf`, B580 |
| 2 | i8 V cp-pair read reuse (two cp blocks from one 32-row read, 8 -> 4 reads), 3 attempts (`if(cp&1)` gate / explicit `cp+=2` unroll / grouped-mad schedule) | v1 21,666 -> 22,812 ns **slower**, the two rewrites also no improvement, all reverted | read count is not the bottleneck; the gate and iteration-carried state perturb IGC scheduling; HW/L2 absorbs the discarded half | same, B580 |
| 3 | K diagnosis (dequant removed, K_DIAG=2) | actually **slower** (32,916) | removing the scale/zp loads changes IGC scheduling/regalloc (observer effect). Load vs dequant cost cannot be separated | same |
| 4 | V transform_8b + V_SCALE_CACHE=0 | 29,687 **slower than scalar** | per-byte global scale reloads eat the transform gain. The two must ship together | same |
| 5 | replace the KQ per-lane K gather with `char16 vload16` (mixed i8, "1b") | instCount 4868 -> 4565 but mov 2185 -> 2275, d32 16 -> 48 | the "fewer messages, more mov" shape is the same one that lost in decode | `sdpa-ocl-pa-i8-by-token-task`, ISA |
| 6 | decode K prefetch | **-3.6%** (slower) | KQ already interleaves independent DPAS chains | `sdpa-ocl-decode-kernel` |
| 7 | decode prefetch distance 16 | +2.21% | the in-flight set overflows L1 | same |
| 8 | V prefetch right before consumption in plain prefill | about 11 us net loss | no compute to hide it behind | `sdpa_ocl.cl` history (`#if 0` comment), measurement HW not recorded |
| 9 | `MAX_BARRIER_V_PREFETCH` | gain never proven, deleted | no individual measurement file | `sdpa-ocl-refactor-2026-09` |
| 10 | 256 GRF at head-72 | **+47% slower** (9.57 -> 14.12 ms) | even with spill removed, the cost of halving the resident threads is larger; LSC message count is the bottleneck | `sdpa-ocl-block2d-gate-relaxation`, B70 |
| 11 | 256 GRF at u4 head-64 | **+70% slower** (1474574702 -> 2502030737) | a gather-bound config hides latency with thread count | `sdpa-ocl-u4-head64-page-read`, B70 |
| 12 | u4 head-64 `PER_WG_KEYS=4` (minimum message count config (h)) | 397.9 M -> 427.7 M ns **slower** | sg_per_wg 16->8 halves the thread count | same |
| 13 | share the 4x redundant u4 V dequant through SLM | dropped (unmeasured) | -24% ops vs WG 4->3 (-25% occupancy), blind bet | same |
| 14 | force `Q_2D=1`, `A_2D=1` (prologue/epilogue) | +0.35% (meaningless) + **wrong-result risk** because there is no fixup | outside the loop | `sdpa-ocl-block2d-gate-relaxation` |
| 15 | decode K as `vload16` per-lane rows (token-major) | instructions 1076 -> 980 but wall-clock unchanged to slightly slower | scatter 16 lines vs block 8 lines | `pa-k-token-major-progress` |
| 16 | kvup `PA_K_SGS_PER_WG` (partitions that share the same sector in one WG) sgs 1/2/4/8/16 | no gain (sgs=2 is +479 ns) | Xe2 does not merge sub-line writes from different threads that use the same L1 | `pa-kvup-token-major-store-sector-floor`, B70 |
| 17 | kvup loop compile-time unroll + predication | 3700 -> 3618 (2%) | not the limiting factor | same |
| 18 | kvup partition count 16 -> 8 (h_sub>=2) | the ratio improves (0.90) but in absolute terms 4337 vs 3700 is **slower** | 2x serial work per thread; the production baseline is d-major@parts16=3353 | same |
| 19 | kvup partition-tiled page layout `[K/16][16][8 B]` | rejected on scope (not tried) | would require rewriting the DPAS operand assembly in decode/MIXED | same |
| 20 | lane-varying loop variable in the u4 token-major writer | 1.8x slower | `token_vals[]` lost its registers | `pa-kvup-u4-token-major-writer` |
| 21 | `skip` guard inside the helper body | 18.7 us (2.3x slower than unsplit) | inner loop unroll lost | same |
| 22 | keep the paired Kc DWORDs across parities (u4 MIXED) | 177,716 vs 158,661 ns (**about 12% worse**) | a long-lived private array causes spill/register pressure | `sdpa-ocl-mixed-exact-complete`, B70 |
| 23 | MIXED `KQ_FAST`, `KQ_TRIM` | 172,310 / 157,932 (worse), removed | compiler-level cause unproven | same |
| 24 | before rung 3, "the basic suite is green, so rotate is OK too" | rotate corrupted all 512 pairs | the default suite does not observe the cache contents | `pa-k-token-major-progress` |
| 25 | `16b_32r16x1c` as a drop-in for `2x 16r` | undetermined (invalid because the probe returned all 0) | sentinel check missing | `sdpa-ocl-tk32-bug-hunt` |
| 26 | `V_I8_PAIRED_READ` on the PA path | **forbidden** | consecutive key groups are on non-adjacent pages (`block_indices`) | `sdpa-ocl-pa-i8-by-token-task` |
| 27 | at head-72, `DKS_ACTIVE` enlarges the spill of the scalar fallback (head 48/96) | static 2688 -> 16064 (DKS_ACTIVE 8/6/5) | **not measured on device (as of 2026-08-14)** | `sdpa-ocl-block2d-gate-relaxation` |
| 28 | kvup writer slots as `half16` with a runtime subscript (instead of selects over a static array) | no scratch, but GENERATE h256 6.15 vs 5.35 us, h128 4.49 vs 4.35; PREFILL 2 blocks 31.4 vs 13.7 us | 353 indirect `r[a0]` accesses | `pa-kvup-writer-igc241-spill-task`, B70 IGC 2.41 |
| 29 | kvup head-256 store rotation by a select per store / barrel rotate / `switch` over 8 static orders | h256 5.03 / 5.10 / 5.14 us, but h128 4.53 / 4.63 / 4.84 and h512 8.46 / 7.48 / 8.22 (uchar8 index, ungated: 5.03 / 4.55 / 7.32) | select chains and a `switch` cost more than one indirect read | same |
| 30 | kvup store rotation at every head size | h128 +3..8%, h512 +5%, h256 2 kv heads +1% | fewer writers per line; the rotated read is not free | same |
| 31 | 256 GRF / IGC knobs for the writer's `TPM=576` | private memory unchanged | a runtime-indexed private array, not register pressure | same |
| 28 | f16 V-read merge `16r16x2c` (VTune-based) | **small gain** (not a failure): 524.34 -> 521.01 ms (-0.64%), MD5 identical; SBID stall 43.9–45.5% -> 41.5–43.5% | single-workload case, no guarantee for other GPUs/models (MEASURED) | chapter 04 §4.1.1, chapter 05 §5.5.3, B70 |
| 29 | workgroup order with the KV-group heads innermost (concurrent workgroups share K/V tiles), head 512 PA PREFILL | +5.2% / +4.1% at 8K / 32K, knee unchanged | the knee is miss latency per workgroup, not L3 reuse across workgroups | `sdpa-ocl-head512-prefill-task`, B70 1800 MHz |
| 30 | 32-row x 2-block K (`16b_32r16x2c`) / V (`transform_16b_32r16x2c`) demand reads, 4x fewer messages for the same lines (head 512) | +3.8% / +14% / +9% (K / V / both, 8K, one run, no spill; correctness not checked) | message count is not the cost; register payload and schedule change | same |
| 31 | address prefetch of this tile's V at a 64 B stride (head 512), `L1UC_L3C` / `L1C_L3C` | +17.0% / +15.0% at 8K | 4 requests per 256 B line (§7.6) | same |
| 32 | next-tile K address prefetch (head 512), 64 B / 256 B stride | +15.7% / +4.0% at 8K | K misses are not worth their requests | same |
| 33 | `PREFETCH_V_L3` in place of `PREFETCH_V` at `d_max` <= 256 | +1..+5% (head 256 full layers -1%) | the K/V of one KV head fits L3 | same |
| 34 | Q_slm as `ulong4` (`WIDE_READS` bit 4) at `d_max` 128 / 256 | d128 +1.4..+4.0% alone, +3.8..+16% with the K pair (qwen3-8b, phi-3.5); d256 +0.1..+2.2% | 8 deinterleave movs per Q read; the loop is not memory-bound there (§5.4) | `sdpa-ocl-wide-reads-other-heads-task`, B70 1800 MHz |
| 35 | K pair + Q_slm on a 128-key sliding-window layer (`d_max` 64, gpt-oss-20b) | **+10..+12%** at 1K-32K (full layers -2.8..-4.7% at 4K-32K) | K/V of a small window stay in cache, the movs dominate; shipped as 0 for `d_max` 64 with a window | same |
| 36 | K pair alone at `d_max` 64 on gpt-oss-20b (sink input) | full layers +1.2..+1.8%, window layers 0..+0.7% (llama-3.2-1b: -0.4..-0.7%) | schedule effect of the kernel variant; not explained | same |
| 37 | V pair at `d_max` 256 (gemma-3-4b) | +0.6% full layers, -0.1% window layers | no gain; the only tiling below 512 where it exists | same |
| 38 | `intel_sub_group_block_write_ul4` to stage Q_slm | **wrong Q** on the dynamically padded head-64 rows `basic/29`-`34` | not identified; per-lane `ulong` stores of the same layout are correct | `sdpa-ocl-d512-wide-reads`, B70 |
| 39 | f16 d-major K page: one 16-row transpose per depth tile instead of the 32-row pair read (256 GRF) | 508 vs 460 us (qwen3.5-9b 8K) | twice the transpose messages; Send stall 7.8% vs token-major 2.4% | `sdpa-ocl-slower-than-micro-task`, B70 1800 MHz |
| 40 | partial-page zeroing of the transposed K rows behind `if (kp_rows < 16)` | 565 vs 508 us | the branch made IGC copy k_raw (16 phi `mov`s per depth tile); ANDing in the extraction is free | same |
| 41 | 32-row transpose pair read at 128 GRF (`d_max` 64 table tiling) | spill 4,096 B, 497 vs 413 us (llama-3.2-1b), gpt-oss 932 vs 731 | 16 dwords per parity live across the depth loop | same |
| 42 | i8 MIXED K page prefetch (`PA_MIXED_PREFETCH_K`) | 0 to +1.6% on 5 models | the depth-tile re-reads already hit L1 | same |
| 43 | `transpose_32b_16r16x1c` (one read for a channel pair, 16 dwords) | not declared in OpenCL C on IGC 2.41 | only in libigc's builtin table | same, ocloc |

## 13. Uncertain/stale items in this chapter (need checking)

- The claim in `sdpa-ocl-int8-perf` that "doing the K dequant in float beats half" (2026-07-05, ISA + the user's measurement) disagrees with the current code: `k_tile_i8_b2d` (`sdpa_ocl_qk_load.cl:~275`) computes the bias trick `as_half(0x6480 ^ byte)` in half and uses float only for bf16 -> possibly superseded by later changes. Content in `03-numerics-softmax-quantization.md`.
- The two "Performance opportunities" items in `docs/sdpa_ocl.md` are unmeasured candidates (decode `% 16` page rule, MIXED V page lookup 2 times, etc.).
- The device label is missing for the decode prefetch measurement (§7.2). It is dated after the B70 migration (2026-07-05), so it is estimated to be B70, but ASSUMED.
- The tk32 override-only bug (`sdpa-ocl-tk32-bug-hunt`) is UNRESOLVED (2026-09-25: not reproduced in the default config, masked by `MICRO_MATH=1`). Ordering around the split barrier was a suspect, but there is no evidence.
- The `MAX_BARRIER_V_PREFETCH`, `DKS_ACTIVE`, `SDPA_OCL_PA_CUR_*` toggles in `sdpa-ocl-block2d-gate-relaxation` were deleted by the refactor (confirmed in code): the env names in the note are a historical record.
