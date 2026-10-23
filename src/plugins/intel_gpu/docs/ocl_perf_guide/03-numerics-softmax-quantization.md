# 03. Numerics design: online softmax, masks, quantization dequant, accuracy verification

Scope: in `src/plugins/intel_gpu/src/graph/impls/ocl_v2/`: `sdpa_ocl.cl`, `sdpa_ocl_decode.cl`, `sdpa_ocl_mask.cl`,
`sdpa_ocl_qk_load.cl`, `sdpa_ocl_v_load.cl`, `sdpa_ocl_config.cl`, and the writer `pa_kv_cache_update_ref.cl`.
Host jit: `sdpa/sdpa_gen_ocl.cpp`, `sdpa/sdpa_gen_ocl_decode.cpp`. Design doc: `docs/sdpa_ocl.md`.

Notation: **MEASURED** = measured value with a source (note/commit/hardware), **ASSUMED** = code reading or hypothesis. Hardware abbreviations: B580 / B70 = Xe2 dGPU,
DG2 = Xe-HPG; all B580 numbers predate B70 and are treated as stale (`hardware-migration-b580-to-b70` note). Static ISA numbers (instCount, mov count) are
`ocloc -device bmg` results, not time.

Related chapters: DPAS operand roles/tiling → `01-dpas-and-tiling.md`, 2D block read/prefetch/barrier → `02-memory-io-prefetch-barriers.md`,
spill/ISA/profiling → `04-spill-isa-profiling.md`, general verification methodology → `05-methodology-and-pitfalls.md`.

---

## 0. One-page summary (numerics rules)

| Rule | Basis |
|---|---|
| Accumulation (accumulator) is always f32: DPAS `float8` acc, softmax max/sum f32, S*V acc f32. Only the DPAS inputs are f16/bf16 | `sdpa_ocl.cl:365` `float8 A_tile`, decode `S_VEC_TYPE` = `SOFTMAX_ACCUMULATOR_TYPE` |
| Softmax probabilities P are rounded to the input dtype (f16/bf16) and stored in SLM; the denominator sum adds the f32 exp values before rounding | `sdpa_ocl.cl:807-823`(PACK_SOFTMAX8), decode `:702-710` |
| log2(e) is folded into the scale and `exp2` is used (prefill). Decode pre-multiplies Q by the scale and uses `native_exp` | `sdpa_ocl.cl:156,807`, decode `:408,702` |
| Fully-masked rows/tiles output 0, not NaN (`l > 0 ? 1/l : 0`, `isfinite(m_new)` guard) | `sdpa_ocl.cl:780,1017` |
| Quantization dequant is affine, so scale/zp can be moved outside DPAS (onto the scores or the probabilities) | §5 |
| The f16 integer widen trick is exact only for an integer zp. A non-integer zp gets rounded at ulp=1 (1024~2048) | §4.5, §10 |
| NaN under a mask: `NaN + -INF = NaN`. NaN in unwritten cache slots is not hidden by the mask, so set it to 0 at the load stage | §2.6 |
| Do not judge kernel correctness by an end-to-end metric (WWB). Compare layer-0 with identical inputs, against the output dtype ULP | §9 |

---

## 1. Online (flash) softmax design — `sdpa_ocl.cl`

### 1.1 State and recurrence (a: what)

For each key tile k0, update the per-query running max `m`, running sum `l`, and accumulated output `A`:

```
m_new = max(m_old, max_k s_k)            // s = raw QK^T (scale not applied) + mask*iscale
alpha = exp2((m_old - m_new) * scale')    // scale' = scale * LOG2E
p_k   = exp2(s_k * scale' - m_new * scale')
l     = alpha * l + sum_k p_k
A     = alpha * A + P @ V
O     = A / l   (epilogue, 1x)
```

- The S tile (`S_tile`) holds **unscaled** raw scores, and the max is also taken in the raw domain (`S_max_slm`). `scale' = scale * LOG2E`
  is multiplied only once, in the exp2 argument (`sdpa_ocl.cl:130-156`). Masks/sinks are in the raw domain, so they are added with `* iscale` (= 1/scale)
  (`:651,700,717`; sink `:163`).
- The default path stores the scaled max (`S_max_tile = m_new * scale'`) and computes `exp2(S*scale' - m_log2)` → 1 fma per argument.
  The `MICRO_MATH` path subtracts the raw max first and then scales (`exp2((S - m) * scale')`), sums the keys in order, and accumulates S*V in `A_tile1` (starting from 0) before
  adding it — to reproduce the rounding of `sdpa_micro` (`sdpa_ocl.cl:781-831, 870-879`, host `sdpa_gen_ocl.cpp:677-688`: PA PREFILL f16,
  ON by default only when there is no sink/alibi/sliding-window/token_type_ids; `SDPA_OCL_MICRO_MATH=0` turns it OFF).
- Reason: the two paths have a **different numerical order**. The default path rounds the `exp2` argument differently, so it is not bit-exact. If bit-closeness to micro is the goal,
  use MICRO_MATH; otherwise use the default. (See the "order-changing" classification in §9)

Key snippet (`sdpa_ocl.cl:780-831` abridged):

```c
const bool ok = isfinite(m_new);                       // m_new = S_max_slm[query] (result of the atomic max across subgroups)
const float m_log2 = ok ? m_new * scale : 0.0f;
const float a = ok ? native_exp2(S_max_tile[qb] - m_log2) : 1.0f;   // alpha
S_max_tile[qb] = ok ? m_log2 : S_max_tile[qb];
float8 exp_tile = ok ? native_exp2(S_tile[mb][qb] * scale - m_log2) : (float8)0.0f;
lsum += exp_tile[0] + ... + exp_tile[7];               // f32 sum, before P is rounded
S_sum_tile[qb] = a * S_sum_tile[qb] + lsum;
vstore4(as_uint4(PACK_SOFTMAX8(exp_tile)), 0, &S_slm[...]);   // P goes to SLM as f16/bf16
// rescale the previous A by alpha (except the first tile):  A_tile[r][cd] *= av
```

### 1.2 pitfalls (d)

| Pitfall | Explanation / basis |
|---|---|
| `-inf - -inf = NaN` | A query row with no valid key up to that tile (early causal, sliding window, fully-masked) has `m_new = -inf`. Forming `S - m` without a guard contaminates S/A with NaN. `ok = isfinite(m_new)` sets the exp argument to 0 and `alpha = 1` (`:777-795`) |
| `first`/`last` flags | `first` gates the alpha rescale, `last` gates the `S_sum_slm` write. When causal/window moves the loop start/end, **both must move together** (`:481-485`). Otherwise the result is corrupted / the sum is never written (causal-bound note) |
| `alpha[]` runtime indexing | Reading a private array with a runtime index makes IGC put it in scratch (llama-3.2-1b MIXED: private memory 128 B + scratch ld/st 2+2 inside the k0 loop). Solved with a select chain over the compile-time `kq_query_blocks` (`:849-861`) — no numerical change, done for performance |
| `native_exp2(-inf)` | When the first valid tile is not `first`, `alpha = native_exp2(-inf - m)` must be 0 so that `A*alpha` stays 0. **ASSUMED**: Xe2 HW exp2 maps -inf→0. (The extreme-value behavior of OpenCL `native_*` is implementation-defined) There is a path exercised by tests (early causal tiles), but no dedicated unit check was found |
| P precision in S_slm | P ∈ [0,1], so no f16 overflow. The denominator is the f32 sum before rounding while the numerator is the rounded P: ~5e-4 relative mismatch. micro has the same structure |
| Single max source | `S_max_slm` is the running max that the subgroups merge with `__builtin_IB_atomic_max_local_f32`. Initial value is -INFINITY (or the sink) (`:319-327`) |

### 1.4 Branchless exp (`SOFTMAX_BRANCHLESS`, PA PREFILL, 2026-10-23)

`float8 exp_tile = ok ? native_exp2(...) : (float8)0.0f;` was compiled by IGC as a goto/join **around the exp2 of every (mb, qb) block**, so
the blocks could not interleave. Replacement (both the `MICRO_MATH` and the default arm):

```c
float8 exp_tile = native_exp2((S_tile[mb][qb] - m_new) * scale);
exp_tile = select((float8)0.0f, exp_tile, (int8)(ok ? -1 : 0));
```

- Bit-identical: `select` returns exactly what the ternary returned. When `!ok` (`m_new = -inf`) the exp2 of `S + inf` / NaN is computed
  and **discarded by the select**, so no Inf/NaN escapes into `S_slm` or `lsum`.
- MEASURED (B70 pinned): switching it off again costs +2.6% (qwen3-8b 8K, head 128); -2.9% at `d_max` 512 (gemma-4-e2b 8K).

### 1.3 Differences in decode (`sdpa_ocl_decode.cl`)

- Keys are 1 per lane (`lane == key`), so max/sum take 2 subgroup reduces (`:667-700`). Each partition normalizes against its own max and
  writes `exp_sums`/`max_logits`, and `pa_sdpa_finalization_stage` merges them.
- Masks with **`SOFTMAX_ACCUMULATOR_VAL_MIN` instead of -INFINITY** (`:627-640`). Reason: a whole partition can be masked (the sliding window starts much later),
  and `MIN - MIN = 0` keeps exp finite (=1); finalization then clears that partition to 0 via `exp(max_logit - global_max)`.
  This is a different convention from the prefill kernel, so do not mix them (because of the partition-merge structure).
- Q is pre-multiplied by `scale_val` (`:408`) → the scores are already scaled. So in decode, mask and sink are not raw. Copying the prefill sink rule
  (`* iscale`) into decode is wrong.
- With 1 partition it writes the output directly; with several it writes `tmp_out` divided by the partition's own sum. This branch condition must be exactly complementary to **finalization's
  `effective_seq_len = seq_len - swa_start_token`** (T4: output-not-written bug for SWA window 240..256, fixed in `03c9121efc`).

---

## 2. Masks

### 2.1 causal / sliding window key-loop upper and lower bounds (a/b/c)

Trusting only the mask and looping over all keys was the root cause: `for (k0=0; k0<k; ...)` + `if (key > query) s = -INF` loads K/V and runs DPAS for tiles beyond the diagonal too, then throws them away.
At q=k=1024, head 128, 32 heads: ocl (query tile 32) causal efficiency 50% vs micro (128) 80% (causal-bound note, arithmetic estimate).

```c
// sdpa_ocl.cl:383-395, 417-434 (abridged)
int causal_k = min(k, query_position_offset + (int)wg_j0 + kq_wg_tile_queries);
int window_k_begin  = max(0, query_position_offset + (int)wg_j0 - SLIDING_WINDOW_SIZE + 1);
const int window_k0_begin = (window_k_begin / kq_wg_tile_keys) * kq_wg_tile_keys;   // round down to tile alignment
for (int k0 = window_k0_begin; k0 < causal_k; k0 += kq_wg_tile_keys) { first = (k0 == window_k0_begin); last = ...; }
```

- The window lower bound is **rounded down to a k0 tile boundary** — the 2D block read and the `S_slm` indexing assume that `key_base` is tile-aligned.
- 0 iterations is impossible: `window_k0_begin <= wg_j0 < causal_k` (note: brute force over 7.76M (query,key) pairs, W∈{1..512}×k∈{16..512}, 0 needed pairs skipped).
- Block-level skip `causal_block_clear` (`:668-682`): if the whole block is inside causal(+window), the per-element predicate is skipped. Subgroup-uniform.
- Measured (B70, fixed 2800MHz, PA prefill q=1024, 32/8 heads, head 128, f16, min ns): without causal_k 3,421,875 → 3,300,312 (**1.78x fewer tiles, but 1.04x in time**),
  window (W=256) 2,266,562 (**1.71x fewer tiles, 1.46x faster**). The tile-count reduction does not translate linearly into time (hypothesis: in-WG reduce/barrier overhead, **unresolved**).
  On the real llama-3.1-8b model the causal_k effect was confirmed by the user; window has no effect there because the model has no SW (compiled out).
- Test coverage gap (note, may have changed since): at the time no case hit sdpa_ocl PREFILL + sliding_window (it was routed to another backend).

### 2.2 causal lower-right alignment (plain SDPA)

In stateless decode (`q < k`), aligning the mask lower-right makes query i see keys `[0, i + (k - q)]`. Without the alignment, at `q=1,k=512`
`causal_k = min(512, 0+32) = 32`, so keys 32..511 are **not even visited** and the denominator is wrong (`decode_40q_40kv_512seq`).

```c
// sdpa_ocl_config.cl:178  (PA already applies the same shift via past_len, so ON only for plain)
#define LOWER_RIGHT_SHIFT(pos) pos + causal_offset       // causal_offset = max(0, k - q)
```
- Applied in 3 places: the `causal_k` upper bound, `causal_block_clear`, and the per-element predicate + window lower bound (all with the same offset, `sdpa_ocl.cl:388,671,740-743`).
- The macro has **no parentheses**: safe only as an operand of a comparison operator. Used inside an arithmetic expression it becomes a precedence bug (latent).
- Host: `jit.make("CAUSAL_MASK_LOWER_RIGHT", config.is_paged_attention ? false : config.causal_lower_right)` (session record `test/SDPA_OCL_CAUSAL_LOWER_RIGHT_SESSION.md`).
  With `=0` the preprocessed output is byte-identical; ocloc inst 1190 → 1194 (+4). User confirmed: all of `*sdpa_gpu_causal_mask*` pass.

### 2.3 Mask kinds (`MASK_KIND`) and OOB

| MASK_KIND | Shape | Handling |
|---|---|---|
| 2 | full 2D `[.,.,q>1,k>1]` | lane=query, tile loaded with a `half16` vload, then `* iscale` (`sdpa_ocl_mask.cl:56-84`) |
| 1 | per key `[.,.,1,k]` | `mask_tile` loaded with lane=key, `sub_group_broadcast` |
| 0 | scalar/broadcast | `HAS_SCALAR_ATTN_MASK`: `s += MASK_TO_FLOAT(msk[0]) * iscale` (`:697-701`) |
| -1 | decided at runtime | branches on `MSK_D2`/`MSK_D3` |

- **Dynamic shape pitfall**: when the mask trailing dim is dynamic, the host **guesses** the kind from the stage (prefill → 2). If the actual runtime mask is `[B,H,1,K]`, the full-2D loader
  reads query rows up to `~990`, overrunning the 1-row buffer by ~15x → on Xe2 `CL_OUT_OF_RESOURCES` (an OOB read crashes instead of returning 0) or garbage→NaN.
  Fix: `mask_query = (MSK_D2 == 1) ? 0 : ...`, row bound `mask_query < MSK_D2`, and on the key side also the `MSK_D3 == 1` broadcast + `mask_key + kk < MSK_D3` (`sdpa_ocl_mask.cl:62-80`).
  Static masks are literals, so the select folds → codegen byte-identical (OOB session record `test/SDPA_OCL_MASK_KIND_OOB_SESSION.md`).
- **Scalar runtime mask**: a rank-0/1-element input was not treated as a "real mask", so its value was ignored and `CL_OUT_OF_RESOURCES` also occurred. Handled with a separate helper (`has_scalar_runtime_attn_mask_input`) +
  `HAS_SCALAR_ATTN_MASK` (session `test/SDPA_OCL_SCALAR_RUNTIME_MASK_SESSION.md`). sdpa_micro keeps this gate (unsupported).
- bool/half mask values are unpacked to float with `MASK_TO_FLOAT*` and then `* iscale` — for bf16, `as_ushort` → bit-decode (§8).

### 2.4 bidirectional (`token_type_ids`, image groups)

The allowed key set is the **union** `(causal ∩ window) ∪ (own image group)`, not a pair predicate. So only the **query-side `[group_begin, group_end)`** is needed
(1 scan per WG, int2 per lane), and the hot loop is the single line `key < gb || key >= ge` (`sdpa_ocl.cl:745-752`).
- Extending the key upper/lower bounds: groups are contiguous, so scanning **only the last query of the WG** is enough (`bidir_scan_end`, subgroup-cooperative `sub_group_reduce_min/max`, `sdpa_ocl_mask.cl:13-46`).
  Clamping the scan to the k0 loop range is **exact**, not an approximation (keys outside the range are either not visited or already -INF).
- `causal_block_clear` stays valid: the image rule only un-masks, and a skip that proves the block is inside causal+window has nothing to un-mask.
- Coordinate systems: `token_type_ids` is **LOCAL (new tokens `[0,q)`)**, `key/causal_k/window_k_begin/bidir_group_*` are **KEY (= query_position_offset + local)**. Mixing them up is the most common source of bugs. Cached tokens can never be in the same image group as a new query.
- GENERATE does not need it (proof): 1 token/subsequence, so the group = itself → plain causal. MIXED needs `kv_cache_update` to run before SDPA to read the group's future keys (nothing asserts this).
- `[B_token | 0]` runtime gate: access only when `token_type_ids_count > 0` (`bidir_active`). Reading an empty tensor is OOB (`sdpa_ocl.cl:199-216`).

### 2.5 qq_bias (speculative tree mask, MIXED only)

For (query, key_spec) pairs inside the new key range `[past_len, past_len+spec_num)`, `qq_bias == 0` gives `s = -INFINITY` (`sdpa_ocl.cl:720-733`). The jit/signature/arguments must all
agree on `HAS_QQ_BIAS && IS_PA && !IS_PREFILL` (at one point the jit was hardcoded to 0 and the arguments shifted). Decode has 1 new token, so it is a 1x1 identity and need not be read (session `test/QQ_BIAS_SDPA_OCL_DECODE_SESSION.md`).

### 2.6 Fully-masked rows, NaN/Inf handling summary

| Situation | prefill (`sdpa_ocl.cl`) | decode |
|---|---|---|
| No valid key in a tile | `ok=false` → exp_tile=0, alpha=1, S_max_tile kept | `VAL_MIN` mask, removed in the partition merge |
| No valid key in the whole row (`l == 0`) | output **0** (`inv_l = (l > 0) ? recip(l) : 0`, `:1017`) | N/A |
| Unwritten cache slot (may hold NaN) | height clamp / forced scale=zp=0 at the load stage. **The mask's `+ -INF` cannot erase NaN** (`NaN + -inf = NaN`) | K: the mask **overwrites** `s` (`s[g] = VAL_MIN`), erasing the NaN. V: probability 0 is a multiply, so `0*NaN = NaN` → V needs the `v_valid ? v_sc : 0` guard (`:788-803`) |
| causal predicate | assigns `s = -INFINITY` (not an addition) | — |

`fmax(lmax, NaN)` ignores NaN, so the max is fine and only the sum becomes NaN — **the symptom shows up late**.
Whether output 0 matches the reference (whether the reference produces NaN) was **not verified (ASSUMED)**.

---

### 2.7 Last-tile-only causal mask (`CAUSAL_MASK_LAST_ONLY`, PA PREFILL, 2026-10-23)

A workgroup's queries all lie in its **last** key tile when the WG query tile divides the WG key tile and both are aligned, so no earlier
tile holds a causally masked key: non-last tiles skip the mask loop and only take the row maxima (same order -> bit-preserving).

```c
#if SLIDING_WINDOW_SIZE
const bool tile_masked = last || (k0 + SLIDING_WINDOW_SIZE <= (int)wg_j0 + kq_wg_tile_queries - 1);
#else
const bool tile_masked = last;
#endif
```

- The window term: an earlier tile still needs the mask while it holds a key the window drops for the workgroup's **last** query.
- Host gate (`add_prefill_pipeline_jit()`): `!has_token_type_ids && wgTK % wgTQ == 0`; `sdpa_ocl_config.cl` `#error` with bidir, attn mask,
  MIXED, lower-right or `wgTK < wgTQ`.
- Without it IGC flattened the per-block `causal_block_clear` branch into 64 compares + 64 selects per subgroup per tile (more with the window).
  MEASURED (B70 pinned): qwen3-8b 8K 7,218 -> 6,977 us (-3.3%; -4.2% at 32K); phi-4-mini 8K 4,813 -> 3,753 us (micro 4,191), 32K 68,485 -> 53,489.
- **Trap: a window no prompt reaches.** phi-3.5/phi-4 carry the HF `sliding_window` 262144 into the PA op, so they count as SWA models:
  `MICRO_MATH` is excluded and a `SLIDING_WINDOW_SIZE == 0` gate would drop the fast path (phi-4-mini was 1.12-1.16x micro until the
  window-aware condition above, then 0.87-0.96x). Gate fast paths on what the window actually cuts, not on its presence.
- Test design: the mask rows must be **uncompressed**. A compressed cache relaxes the PA test tolerance from 2e-3 to 25e-3 (u4 75e-3), and
  a dropped window term PASSed there. Negative control (window term removed): `{1000}`/300, `{700,300}`/256, head 96 `{520}`/100, head 64
  `{900}`/200, head 256/512 `{700}`/200 and the sink row FAIL; `{600}`/4096 and the windowless rows PASS.

## 3. Attention sink (gpt-oss)

sink = one extra logit per head, **with a value vector of 0** → it only enlarges the softmax denominator.

- **prefill/MIXED (`sdpa_ocl.cl`)**: **seeds** the online-softmax state. 0 hot-loop changes.
  ```c
  const float sink_raw = SINK_TO_FLOAT(sink_ptr[b0]) * iscale;        // :163 raw domain (= same as mask)
  S_max_slm[qi]   = sink_raw;                                          // :324 running max starts at the sink (1x for the whole WG)
  S_max_tile[qb]  = sink_raw * scale;                                  // :354 same product → the first alpha is exactly 1.0
  S_sum_tile[qb]  = (sg_i_kq == 0) ? 1.0f : 0.0f;                      // :358 exp2(sink - sink) = 1, in only one subgroup
  ```
  Seeding `S_sum_tile` in every subgroup makes the epilogue's `S_sum_slm` sum count the sink `kq_sg_per_wg_keys` times.
- **decode**: injected only in partition 0 (`:685-694, 746-752`): the sink goes into the max in `m_wg`, and `native_exp(sink - m_wg)` into `lw`. Injecting it in every partition makes finalization count it P times.
  `SINK_HEAD(m)` is a leftover-clamped index (so the fake heads of the last WG do not read OOB).
- Host: there was a latent bug where `HAS_SINK_INPUT` was jitted but the kernel did not declare the argument, causing an **arg shift**. Confirmed a 3-way match of jit/signature/push order.
- Performance: the sink is 13 inst out of 3529 inst (**MEASURED**, ISA A/B `u4_bychannel_1d_sink` vs `u4_bychannel_1d`, B580). gpt-oss was slow not because of the sink but because of the u4 head-64 K/V load (`sdpa-ocl-u4-head64-page-read`).
- Test data pitfall: the data is N(0,0.1), so logit≈0.01 and denominator≈kv_len → the sink value must be sized to the denominator, e.g. `log(kv_len) + spread[h%4]`. Otherwise the test passes even with the sink removed (vacuous).
- A negative control that "correctly shows no change": **leaving the sink out of the max** does not change the result (the max is a free stabilization parameter; numerator and denominator scale together). It is required only when the whole of partition 0 is masked.

---

## 4. int8 → f16 conversion trick (`0x6480`)

### 4.1 What / why

Xe2 has no direct byte→half convert. `convert_half16(as_char16(x))` becomes a 3-step staircase of byte deinterleave (`mov :b`), `:b→:w<2>`, `:w→:hf<2>`, i.e. about 3 movs per element.
Instead, it is solved with **one XOR + (later) a half subtraction**.

```c
// sdpa_ocl_qk_load.cl:294-295, sdpa_ocl_v_load.cl:157, sdpa_ocl_decode.cl:120-127
const ushort wbits = (ushort)0x6480 ^ (ushort)(w_byte & 0xFFu);   // w_byte = 2's-complement bits of a signed int8
const half   wide  = as_half(wbits);                              //  == (float)s + 1152.0
```

### 4.2 Bit derivation

f16 = `S(1) E(5) M(10)`, value = `(1 + M/1024) * 2^(E-15)`.

1. `0x6400 = 0 11001 0000000000` → E = 25 → 2^10 = **1024**, M = 0.
2. With E=25 the range is `[1024, 2048)` and ulp = 2^10 / 1024 = **1.0**. So `0x6400 | n` (n ∈ [0,1023]) = **1024 + n**, represented exactly at integer spacing 1.
3. `0x6480 = 0x6400 | 0x0080` → M = 128 → 1024 + 128 = **1152.0**. (`0x6480` is "1152.0h")
4. A signed byte s (2's complement) is the offset binary `s + 128` with **bit 7 inverted**: `u = s & 0xFF`, `u ^ 0x80 = s + 128 ∈ [0,255]`.
   - s ≥ 0: u = s, bit7 = 0 → set → s + 128.
   - s < 0: u = s + 256, bit7 = 1 → cleared → s + 256 − 128 = s + 128.
5. Therefore `0x6480 ^ u` = `0x6400 | ((u ^ 0x80) )` = `0x6400 | (s+128)` → value = 1024 + s + 128 = **s + 1152**. The XOR does (a) the sign→offset-binary conversion and (b) the exponent-field injection at the same time.
   (The `0x80` bit is a mantissa bit, so the exponent field `0x6400` is untouched: a byte is 0..255 → only the low 8 mantissa bits change.)

### 4.3 Example (worked)

| s (int8) | u = s&0xFF | u ^ 0x6480 (=h bits) | mantissa | value | s + 1152 |
|---|---|---|---|---|---|
| +5 | 0x05 | 0x6485 | 133 | 1024+133 = 1157 | 1157 |
| 0 | 0x00 | 0x6480 | 128 | 1152 | 1152 |
| −3 | 0xFD | 0x647D | 125 | 1024+125 = 1149 | 1149 |
| −128 | 0x80 | 0x6400 | 0 | 1024 | 1024 |
| +127 | 0x7F | 0x64FF | 255 | 1279 | 1279 |

Note record: a C program confirmed that `as_half(0x6480 ^ u) - 1152.0 == (float)s` holds exactly for all 256 int8 values (`/tmp/verify_bias.c`, int8-perf note; the file itself is volatile).
IGC compiles this to exactly 2 ops, `xor(:w, 0x6480)` + `add(:hf, 0xE480 = -1152)` (ISA confirmed, note). This is the same lowering as sdpa_micro/gemmstone `planInt8ToHF` (copy_plan.cpp:1201-1220).

### 4.4 How to fold scale / zp

- **Fold the bias into zp** (`zpb = zp + 1152.0h`): per element `(wide - zpb) * scale` = 1 sub + 1 mul; it only pays off when there is no separate bias-subtract add
  (v1, with the bias subtraction kept separate, is 145 inst/62 mov with the adds blowing up; the folded v2 is 130/62 — microbenchmark, B580 static ISA).
- The subtraction result is `s - zp`, so **if zp is an integer it is exact in f16** (`1152+s` exact, `zp+1152` exact, difference ∈ ±255 exact). Only 1 rounding, at the scale multiply.
- Byte extraction is `(w >> (8*bb)) & 0xFF` (shift+mask). `as_char4(w)[bb]` produces a `:b` region deinterleave.
- Build the VNNI dwords directly: 4 bytes → 2 dwords (`K_WIDEN_DWORDS`, decode `:120-127`):
  ```c
  dst[j*2+0] = as_int(((w & 0xFF) | ((w & 0xFF00) << 8)) ^ 0x64806480u);        // byte0, byte1 → half2
  dst[j*2+1] = as_int((((w >> 16) & 0xFF) | ((w >> 8) & 0x00FF0000)) ^ 0x64806480u);
  ```
  The (2i, 2i+1) pairs of a VNNI half2 dword match the byte order as-is → pure dword arithmetic with no sub-register moves.
- For unsigned u8, use `0x6400 | u` (= 1024+u) instead of the XOR. Only signed bytes need `0x6480 ^` (to invert bit 7).

### 4.5 Measured effect and limits

| Item | Value | Source/HW |
|---|---|---|
| K dequant microbenchmark (consumed by dpas, 32 keys, asym) | float 132 inst/77 mov/48 `:b` → bias 100 inst/24 mov/0 `:b`. mov −69% | static ISA, B580 (int8-perf note) |
| Real kernel loop body (head 128) | 2751 → 2534(K) → 2247(K+V) inst, mov 1535→1055→543 | static ISA, B580 |
| Device time (head 128, q=4096, plain int8 prefill, floor ns) | 5,620,520 → 5,406,562(K, −3.8%) → 5,085,625(K+V, cumulative −9.5%); ocl/micro 1.396→1.261 | **MEASURED**, B580, fixed 2800MHz recorded, note. Not re-measured on B70 |
| decode i8 (head128, M=4) | dword widen 4045 inst/2022 mov vs `convert_half16(as_char16)` 4431/2852 | static ISA (docs/sdpa_ocl.md:624-628) |

**Limits (pitfalls)**
- **Exact only for an integer zp.** `zp + 1152` has f16 ulp = 1, so a non-integer zp is rounded to an integer (`np.float16(7.3+1152)-1152 == 7.0`).
  plain SDPA i8 is safe because KV compression stores zp as **i8** (on XMX, `kv_cache_compression.cpp:244`). **Do not use it for int4 (zp = query type) or the PA cache (the writer's non-integer zp).**
  → The PA path spells out `(q - zp) * scale` in half (`PA_DEQ`, `sdpa_ocl_config.cl:256`), and decode removes the bias **separately, in float** (§5).
- bf16 has no such trick (§8). u4 uses a 1024 base (§4.6).
- If the pairing of `K_WIDEN_BIAS` and `zp` breaks, the result is silently wrong: at a level the unit-test threshold (6e-3) does not catch (§5.3).
- The note's wording "denormal reinterpret" is inaccurate: the result is an ordinary normal number (E=25).

### 4.6 u4 nibble

- `as_half(0x6400 | n) == 1024.0 + n` (n ∈ [0,15]), bias = **1024** (`K_WIDEN_BIAS` u4=1024, i8=1152; decode `:115-119`).
- K uses **adjacent** packing (byte b = channels 2b, 2b+1; forced, because the writer's `NUM_K_HEAD_SIZE_PARTITIONS` splits channels across WGs, so the two channels of one byte in different WGs would race). It matches the (2i, 2i+1) pairs of a VNNI dword → natural depth order.
  ```c
  // decode :129-139  K_WIDEN_U4_DWORDS: byte → dword(half2) = (lo nibble, hi nibble)
  as_int(((p & 0x0000000Fu) | ((p & 0x000000F0u) << 12)) | 0x64006400u)   // p = w >> (8*b)
  ```
  The hi nibble (bit4..7) is moved to bit16..19 (low mantissa bits of the second half) with `<< 12`, and the `0x6400` exponent is ORed in → the two halves = (1024+lo, 1024+hi).
- V uses **split** packing (byte b = dim b and dim b+PV): `transform_8b_32r16x1c` has lane == byte column, so with adjacent packing lane c would own both dim 2c and 2c+1 and the DPAS N axis could not be expressed.
  With split packing, reading via `V_TILE_COL(cd)` gives lane c = dim `cd*16+c` (for both the low and high nibble) → the S*V tile loop/output store stay unchanged. The nibble is selected with `cd < V_READS ? (x & 0xF) : (x >> 4)` (decode `:830-836`).
  V uses a plain widen without the bias trick (V zp is per element; f16 ulp@1024 = 1 vs a 0..15 range).
- u4 K in sdpa_ocl MIXED: the A operand is K (lane == head dim), so a byte column of adjacent packing becomes a channel **pair** → no lane-local solution. **Permute the DPAS depth (contraction) axis**:
  `PA_K_U4_CHANNEL(db, L) = win + 2L + par` (`win = (db>>1)*32`, `par = db&1`, `sdpa_ocl_config.cl:278-280`). Tile db (even) takes the even channels of a 32-channel window, an odd tile the odd channels.
  Depth is the contraction, so permuting A and B identically is free. **Only Q pays**: in the 1 SLM staging per WG, `q_pack[j] = par ? ((a>>16)|(b&0xFFFF0000)) : ((a&0xFFFF)|(b<<16))` (`sdpa_ocl_qk_load.cl:58`).
  `DKS_ACTIVE` is grouped in pairs, so on u4 it is always rounded up to even. Verification: `test/check_u4_offsets.py` (symbolically executes the .cl dword expressions to check that Q staging puts at element i the same channel as K lane i, 9 heads × 3 kv-heads, **all 5 negative controls detected**).
- u4 nibble selection is **lane-uniform** (tile parity), so it folds into the shift amount (`U4_NIBBLE_SEL`, `qk_load:400-407`). The writer clamps to `[0,15]`, so there is no sign/CHAR_MIN.

### 4.7 What u4 "exact MIXED" means precisely

"exact" does **not** mean that the dequant arithmetic is exact. Meaning: in MIXED (chunked prefill / prefix caching) the cache `[0, past_len)` is read with u4 dequant,
while **the new rows of the current chunk `[past_len, k)` are read from the unquantized raw f16 `Kc/Vc`** (the current row in the compressed cache is not equal to raw, so going to the cache is wrong).
Implementation: `PA_CUR_KV_F16`; when a k0 tile crosses `past_len`, **the tile is cut** at `k_chunk = past_len - k0` so that one iteration reads only one source (`sdpa_ocl.cl:473-476`),
keeping the fixed DPAS tile shape but masking the rows after `k_chunk` (`k_mask`: `key < k0 + k_chunk`, `:642-646`), then `k0 += k_chunk`.
- The old `GRAN=0` (split at 16-key page granularity) gave **wrong answers** on regression `/0`,`/1`. Now only the exact split exists.
- An odd Kc offset cannot use a dword block read → if `kc_dword_ok` is false, raw Kc scalar fallback. Falling back to the cache here gives wrong answers.
- Performance (**MEASURED**, cliloader average kernel time during a llama-3.2-1b run, measured by the user): OCL 145,603 ns vs micro 147,825 ns (−1.50%). Sub-experiments — SV_TRIM adopted (151,677 vs 158,422), V_PREFETCH adopted (145,603 vs 151,866, −4.12%),
  paired Kc DWORD storage 177,716 (about 12% worse, removed), KQ_FAST 172,310 (worse, removed), KQ_TRIM 157,932 (worse, removed). (Some knobs were later removed in a refactor)

---

## 5. Dequant identities — scale/zp outside DPAS

Dequant is affine (`x = sc * (q - zp)`), so factor it along the multiplication structure. Where it folds depends on which operand has the head dim on its lanes
(→ `01-dpas-and-tiling.md` operand role table). **sdpa_ocl and decode have opposite roles** (prefill KQ: A=K,B=Q / decode KQ: A=Q,B=K), so do not copy another kernel's algebra; re-derive it with the kernel's own mapping.

```
BY_TOKEN K  (decode): S[key] = sc[key] * (sum_d Q[d]*q[key][d]  -  zp[key] * sum_d Q[d])
BY_CHANNEL K(decode): S[key] = sum_d (Q[d]*sc[d]) * q[key][d]  -  sum_d (Q[d]*sc[d]) * zp[d]
V (BY_TOKEN)        : O[d]   = sum_key (P[key]*sc[key]) * (q[key][d] - zp[key])
softmax denominator : sum(P)  — the V scale belongs to the values. Putting it in the denominator is wrong.
```
(`sdpa_ocl_decode.cl:22-27`)

### 5.1 Where to fold

| Term | Location | Reason |
|---|---|---|
| BY_TOKEN K `sc[key]`, `zp[key]` | score post-processing (per-lane scalar, lane == key, no broadcast) | decode: `s = sc*(s - (zp+bias)*q_sum)`, `:613-624`. `q_sum = sum_d Q[d]` is 1 reduce per head (float, `:414-424`) |
| BY_CHANNEL K `sc[d]` | multiplied into **Q (A operand)**: `qv = q_reg * k_sc` (half) | lane == head dim, so it is a per-lane scalar. It differs per page, so it is rebuilt inside the g loop (`:557-566`) |
| BY_CHANNEL K `zp[d]` | subtract a single constant `k_corr[g][m]` from the scores (key-independent, 1x per page·head) | `k_corr = sum_d (Q[d]*sc[d]) * (zp[d] + K_WIDEN_BIAS)` (`:480-497`) |
| V `sc[key]` | per-lane multiply into **P (A operand)** (`pv *= v_sc`) | P already has lane == key. Multiplying into V would need a broadcast onto the head-dim lanes |
| V `zp[key]` | subtracted in the V B operand (`convert_half16(...) - vzp`) | zp varies along the DPAS depth (key) axis, so it is the only one that needs a broadcast; the lane index is a compile-time constant, so it folds into the source region and no shuffle is generated |

### 5.2 Safeguards (NaN / OOB)

- The comp (scale/zp) of keys past seq_len was never written and may be NaN. On the K side the mask overwrites `s`, so no guard is needed; **on the V side `0 * NaN = NaN`**, so `v_valid ? v_comp : 0` forces **both scale and zp** to 0 (`decode :788-803`). Then the widened byte is finite → probability 0 erases the contribution.
- `k_comp_per_key` in plain/PA prefill: `sc_key < k ? scale : 0`, zp is `1152` (bias only) — out-of-range keys become 0 through `scale=0` (`qk_load.cl:105-114`).
- Block reads clamp the page height to `PA_PAGE_ROWS`: because NaN in unused slots **survives** even in masked scores (`qk_load pa_k_tile_b2d16` comment).

### 5.3 f16 rounding rule for BY_CHANNEL `k_corr` (where it goes silently wrong)

The value DPAS sees is the product `Q*sc` **rounded to f16**. The bias (1152/1024) sits inside the B operand as `q+bias`, and cancels exactly against the `bias` term of `k_corr` only for **that same half**.
Computing `k_corr` as a float product leaves a residual: worst-case error 7.21e-4 → 1.68e-3 (2.3x, bias/signal ratio × half ulp). It is below the 6e-3 threshold, so **the tests pass and do not protect it** (docs/sdpa_ocl.md:630-634; measurement hardware not recorded, probe/gtest values).
→ Rule: compute the term that undoes the bias from the half value DPAS actually saw. Folding `zp` into f16 rounds the writer's non-integer zp, so decode removes the bias from the score in float (`:620` `zp_f32 + K_WIDEN_BIAS`).

### 5.4 Writer and comp layout (numerics view)

`pa_kv_cache_update_ref.cl:103-136`:
```c
diff = (max == min) ? 0.004 : (max - min);              // guard range 0 (avoids division by 0 / inf scale)
// u4:  scale = 15/diff,  zp = -min*scale                // non-integer, [0,15] clamp, unsigned
// i8:  scale = 255/diff, zp = -min*scale + CHAR_MIN
q = clamp(convert_int_rte(x*scale + zp), 0, 15)          // round-to-nearest-even
comp[0] = 1.0/scale; comp[1] = zp                        // the writer stores 1/scale → the value read is the multiplier itself
```
- scale/zp are computed in float and then **narrowed to half** (`(INPUT1_TYPE)`). The prefill helper narrows to half and then quantizes, while the requantize helper keeps float: when replacing one, **preserve the promotion rules of the replaced arm, expression by expression** (`range = (max==min)?0.004:(max-min)` subtracts in half and then widens, `fabs(max*0.1f)` is float).
- BY_CHANNEL **requantizes** the whole block (double rounding) whenever a new token entering the block widens the range. Skipping it when the range does not widen is a follow-up optimization (not implemented).
- **requantize pitch bug (unfixed, Known issue)**: the i8 BY_CHANNEL requantize at `pa_kv_cache_update_ref.cl:297` ignores `in_data_pitch` (`j*K_HEAD_SIZE*KV_HEADS_NUM` stride) → with a padded key input it misreads new tokens j≥1 within a block. Found in the T7 review, documented only.

---

## 6. In-place VNNI-stride dequant of V

The DPAS B operand is VNNI-2 (`half2` pairs contiguous along the depth axis). Reading V with `transform_8b_32r16x1c` gives lane == head dim, with each uint holding 4 keys as bytes.
Extract the bytes from this uint with shift+mask and widen with `0x6480 ^` → `deq4 = wide4 - zpb4` → `as_int(deq4.lo)`, `as_int(deq4.hi)` are already dwords in VNNI order (key (2i,2i+1)):

```c
// sdpa_ocl_v_load.cl:155-170  (f16 path)
const half4 wide4 = (half4)(as_half((ushort)(0x6480 ^ ((w >>  0) & 0xFFu))), ..., as_half((ushort)(0x6480 ^ ((w >> 24) & 0xFFu))));
const half4 deq4 = wide4 - zpb4[u];
vb[cd][u*2+0] = as_int(deq4.lo);  vb[cd][u*2+1] = as_int(deq4.hi);
```
- The scale is already folded on the pA side → V dequant is only widen + zp subtraction (`v_i8_comp_fold`, `v_load.cl:114-133`).
- Microbenchmark: the VNNI B-operand pack is the IGC floor — reinterpreting the whole `as_int2(deq4)` and splitting into `as_int(.lo)/as_int(.hi)` are identical (97 inst / 35 `:uw`); it cannot be reduced through source form (v_dequant_bias2_dpas.cl, static ISA, B580).
- The V step gained more than the K step: floor 5,406,562 → 5,085,625 ns (V, −5.9%) vs 5,620,520 → 5,406,562 ns (K, −3.8%) (B580, int8-perf note). Because sv_key_blocks=8, V dequant runs more often per k0.
- Profile (head 128, after the K+V bias change, static ISA B580): S*V is 46% of the loop body (1027/2247 inst); the remaining movs are 64 `pA*vs_c` VNNI `.16` reassemblies, the bias-trick shr/xor, and the VNNI B-pack. A scale/zp broadcast from a constant lane folds into the source region of the consuming op, 0 movs (fact-checked) → "sharing the scale" is not a lever.
- A `sub_group_broadcast` from a constant lane is free, but **a runtime lane is an indirect mov**. Also, reading per-key scale/zp inside the depth/key loop is lane-uniform, so it becomes a SIMD-1 load, 128~256 per k0 → always hoist (docs/sdpa_ocl.md:512-518; the MIXED kernel's load count was dominated by scale/zp: `d16u32` 272 vs data `d8u32` 128).

---

## 7. Dynamic quantization and cache writes (numerics boundary)

- **`dynamic_quantize` head-size cap (T5)**: `Validate()` in `kernel_selector/.../dynamic_quantize_kernel_opt_kv_cache.cpp` rejects `input_dims.back().v > 256` → at head 512 the ref kernel is selected, and
  **the ref kernel cannot append**, so the KV cache breaks and the q=1 iteration is wrong. Suspecting the sdpa_ocl V `_8b_32r16x4c` read was a **misdiagnosis**. Fix: 256 → 512, upstream PR #38466. Lesson: **first compare against `TEST_USE_SDPA_OCL=0` (micro)** — if micro is also wrong, suspect the cache producer (quantization) first.
- **bf16 + compressed KV (14 tests)**: they fail in `add_required_reorders` before SDPA selection because `dynamicquantize` has no i8 layout. Because of this, the bf16 compressed branch of sdpa_ocl is unreachable in real models and untested (T10).
- **The PA compressed path assumes f16**: `pa_v_comp_fold` uses `as_half8(pA)`. bf16 PA is only blocked by `PagedAttentionOpt::validate_impl()` (f32/f16 only), while `sdpa_ocl supported()` lets it through → a host guard is needed (T9 L1, only code-read so far).
- **int4 plain**: zp is the query type (non-integer) and the plain path has no nibble unpack → if dispatched, it would be wrong by more than the zp rounding. Currently `sdpa_opt.cpp` blocks it with `!is_int4_kv`, but `supported()` still compiles it (T9 L2/L3).
- The writer's u4 rule (K BY_CHANNEL + V BY_TOKEN): `execution_config.cpp:312` asserts on a 4-bit BY_TOKEN key → int4 BY_TOKEN K does not exist. u4 only (not i4).

---

## 8. bf16

| | f16 | bf16 (`INPUT0_IS_BF16`) |
|---|---|---|
| Storage | `half` | `ushort` bits (the jitter maps bf16→`ushort`) |
| DPAS | `intel_sub_group_f16_f16_matrix_mad_k16` | `..._bf16_bf16_matrix_mad_k16` (same packing: `short8` A, `int8` B, `float8` acc) |
| P store | `convert_half8` | `_convert_bfloat168_as_ushort8` |
| Mask | `convert_float*` | `_convert_as_bfloat16*_float*(as_ushort*)` |
| int8 dequant | `0x6480` trick + zp+1152 | **float dequant → bf16 encode**, no bias folding |
| scale/zp tensors | f16 | **still f16** (same as micro) |

- bf16 has no trick: the only ulp=1 range is `[128,256)` (7-bit mantissa → 128 slots), so the 256 signed byte values do not fit in one binade. Reading the same bits as bf16 gives ~2^74, not 1152.
  A hybrid (f16 widen → float → bf16 encode) was rejected: it cannot fill the DPAS operand with those bits, and it adds converts.
- **Use `as_char((uchar)b)`**: `(char)(uchar)b` is OpenCL `convert_char`, so 128–255 **saturate to 127** (silent wrong answer).
- Scalar V load: **no numeric `ushort→half` conversion**; copy the bits with `as_ushort(V[...])`.
- With the `INPUT0_IS_BF16` branch, the f16 preprocessed output stays byte-identical. `MICRO_MATH` is for f16 PA prefill only, so it is not enabled for bf16.
- Verification: the user confirmed that `*SDPAWithKVCacheTest.MultipleIterationStateful*bf16*` passes compressed/uncompressed. Performance numbers not measured (only the goal was agreed: ≥ sdpa_micro bf16).
- `SDPAOclGenerator::supported()` had already been letting bf16 through, but **the kernel arithmetic was treating the 16-bit payload as f16** — passing the gate ≠ arithmetic support.

### 8.1 scale dtype (T8, `2cc6103678`)

The runtime scale tensor can be bf16. `SCALE_DATA_T` was hardcoded to `half`, so the bf16 scale bits were read as half: **error 3.5546** (f16 0.0003, bf16 const 0.0021).
Fix: `runtime_scale_layout()` reads the dtype; `SCALE_DATA_T` = half|float|ushort + `SCALE_IS_BF16`; `SCALE_TO_FLOAT`: bf16 uses `as_float(((uint)(x)) << 16)`, the rest `convert_float` (`sdpa_ocl_config.cl:55-57`). Unsupported dtypes are rejected by `supported()`.
- The same bug exists in `sdpa_micro` (`SCALE_DATA_T half`) and in `sdpa_opt.cl:154` (in PA, `SCALE_TYPE = INPUT3_TYPE` → reads an f16 scale as int32): documented only, not fixed (in real models the PA scale is a Constant, so unreachable).
- Proof: negative control (stash of the 3 fixed files) → 4 bf16 tests out of the 8 new tests FAIL; after the fix 8/8 PASS (B70, user confirmed).

---

## 9. Accuracy verification methodology

### 9.1 Test data hides errors

| Pitfall | Symptom | Countermeasure | Basis |
|---|---|---|---|
| PA harness N(0, 0.1), same seed → Q=K=V same array | logit ≈ 0.08, softmax nearly uniform. **Even reading Q/K completely wrong changes the output by only 0.003 (tolerance f16 0.002)**, while a V misread shows up large at ~0.1 | Always add a copy case that sharpens the softmax with `logit_scale_gain=128` (constant scale) or `runtime_scale_multiplier` (runtime scale, Xe2+) | T7: Q-only stride error sharp 0.068 vs plain 0.003 |
| gtest LCG power-of-2 period | `InputGenerateData(-1,2,32,seed)` → range*res = 64 = head size → **every row identical**, identical logits, softmax ignores the scale → PASS even with buggy code | Use a resolution that is not a divisor of the head size (31, 37, 1000); **replay the LCG in 10 lines of Python** to confirm the logit spread/bug effect before handing it over; never the same seed for Q,K,V | T8: spread 0.00(res 32) vs 13.9(res 31) vs 17.1(res 1000), misread error 0.85 vs 0.000 |
| sink value smaller than the denominator | passes even with the sink removed | sink = log(kv_len) + spread | §3 |
| 6e-3 threshold | does not catch the bias-cancellation error (2.3x) | record the error value and compare with previous values | §5.3 |
| reference built from **dequantized** k_log/v_log | the quantization error cancels, so the u4 error is 2.5e-5..5.3e-4, the same band as i8 | do not loosen the threshold | decode-u4 note |

### 9.2 layer-0 identical-input oracle

In a chained network, **only the ops of the first layer** see identical inputs under the two configurations. Later layers receive already-perturbed inputs and measure `kernel error + propagation`.
minicpm4-0.5b (24 PA layers, f16 output, relL2 per (token, head), B70):

| pair | L0 mean | L0 max |
|---|---|---|
| pwk8 vs micro | 1e-6 | 5.07e-4 |
| pwk4 (default) vs micro | 2.5e-5 | 6.82e-4 |

The output is f16, so its own representation granularity is ~4.9e-4 relative → **max ≈ 1.0~1.4 ULP, mean ≈ 0.05 ULP**: a level impossible for an indexing/coverage defect. The later `3e-5(L0) → 7e-3(L1) → 5e-2(L2) → 0.08 plateau` is a 1600x amplification, and
the correct and the buggy configurations reach the same plateau → no information. Self-consistency check: being causal, tokens `[0, tile)` depend only on themselves → they must be bit-identical under both configurations, and the 24-layer mean was 0.0000 (proof that the dumps are comparable).
**VERDICT: the WWB −0.070 of minicpm4-0.5b is not an sdpa_ocl bug (do not re-investigate).** WWB similarity moves by ±0.05 on last-bit changes → **do not use it as a kernel regression gate**; use a per-op exact reference (gtest).

### 9.3 Toggle classification: bit-preserving vs order-changing

| Class | Example | Meaning of "no change" |
|---|---|---|
| bit-preserving | `SDPA_OCL_KV_2D=0`, `_Q_2D=0`, `_A_2D=0`, `_BLOCK_SKIP=0`, `_DKS_ACTIVE=0`, `_256GRF=1` (the extra depth tiles of DKS are exactly +0.0 in f32) | the null result a correct kernel predicts. **Not evidence** |
| order-changing | `SDPA_OCL_KQ_TILE_KEYS/QUERIES`, `_PER_WG_KEYS/QUERIES` (k0 split → accumulation order) | even if it moves the metric on a sensitive model, it is just **a re-roll of the dice** |

"Only the tiling knobs move the score" is not the signature of a defect but the signature of a correct kernel (7 runs wasted). Classify a toggle first, before believing it. (Some knobs were later removed in a refactor.)

### 9.4 Attribution

- Do not attribute a failure from a one-sided control: the A/B must differ in exactly one thing (T5 misdiagnosis: `TEST_USE_SDPA_OCL=0` was not run).
- Pair a prediction for a new guard/gate with **a negative control on the old code** (T7: contrary to the prediction, every 16B padding case passed on the old gate → only distinguishable after adding sharp data).
- Confirm "which kernel actually ran" with cliloader/`get_kernels_dump_info()` (when a gate rejects, execution silently falls through to another backend that is already correct → green ≠ proof).
- set-equality argument: the FAIL→PASS set between a run with only the switch on (writer flipped only, no reader) and a run with the reader toggle also on matches the gate's allowed set exactly.
- Pre-check scripts for offset/nibble agreement: `test/check_by_channel_offsets.py` (4-way match of writer/test fill/reader 2D/gather + comp reads never leave the comp region), `test/check_u4_offsets.py`, `test/check_u4_page_read.py` (set and element equivalence of the 1D `uc16` page read vs the scalar gather, confirms compile-time-constant indices). All in Python, with negative controls.

---

## 10. Numerics pitfall table (summary)

| # | Pitfall | Symptom | Prevention | Basis |
|---|---|---|---|---|
| 1 | `-inf - -inf` | NaN propagates through the row/tile | `ok = isfinite(m_new)` guard, alpha=1, exp_tile=0 | `sdpa_ocl.cl:777-795` |
| 2 | whole partition masked (decode) | exp blow-up/NaN | `VAL_MIN` instead of `-INF`, finalization removes it via `exp(max_logit-global_max)` | decode `:627-640` |
| 3 | NaN in unwritten cache slots | not erased by the mask `+ -INF` | clamp the load height, force scale=zp=0, V probability 0×NaN | `qk_load`, decode `:788-803` |
| 4 | folding +1152 into a non-integer zp | zp rounded to an integer | PA/u4/decode separate the bias in float | §4.5 |
| 5 | `k_corr` in float | error 2.3x, tests still pass | use the f16-rounded `Q*sc` | §5.3 |
| 6 | `(char)(uchar)b` | 128-255 → 127 saturate | `as_char((uchar)b)` | bf16 session |
| 7 | V scale in the denominator | output scale error | fold the scale into P; the denominator is sum(P) | decode `:26-28` |
| 8 | sink in every subgroup/partition | denominator counted multiple times | only `sg_i_kq==0` / `partition_idx==0` | §3 |
| 9 | sink domain | scale mismatch | prefill: raw(`*iscale`), decode: Q is pre-scaled | §1.3 |
| 10 | applying only the causal upper bound | wrong `first/last` | align `first`/`last` with the loop bounds | §1.2 |
| 11 | lower-right not applied | most keys unvisited in q<k decode | `CAUSAL_MASK_LOWER_RIGHT` in 3 places | §2.2 |
| 12 | guessing the dynamic mask kind | 1-row mask OOB → `CL_OUT_OF_RESOURCES`/NaN | `MSK_D2==1`/`MSK_D3==1` clamp | §2.3 |
| 13 | LOCAL vs KEY coordinates | wrong bidir mask | only `token_type_ids` is LOCAL | §2.4 |
| 14 | runtime scale dtype | bf16 scale misread (3.55) | `SCALE_TO_FLOAT` + layout dtype | §8.1 |
| 15 | writer: range 0 | division by 0 | `0.004`, rte rounding, clamp | §5.4 |
| 16 | requantize pitch | j≥1 misread with padded input | **unfixed** (`:297`) | §5.4 |
| 17 | dynamic quantize 256 cap | at head 512 the ref kernel cannot append → cache corruption | upstream PR #38466 | §7 |
| 18 | test data | hides errors | sharp copy case, LCG replay | §9.1 |
| 19 | judging by an end-to-end metric | misdiagnosis from chaotic amplification | layer-0 + ULP | §9.2 |
| 20 | switching the accumulator | switching to an f16 acc grows the error with length | keep the f32 acc (this codebase is all f32, no experiment changing it — **ASSUMED**) | — |
| 21 | a window no prompt reaches (262144) | an "SWA" model loses `MICRO_MATH` and window-gated fast paths | gate on what the window cuts | §2.7 |
| 22 | compressed-cache test tolerance (25e-3 / u4 75e-3) | a missing window mask PASSes | uncompressed rows + negative control for mask changes | §2.7 |
| 23 | `ok ? exp2(..) : 0` | IGC goto/join per block, no interleave | exp2 then `select()` (bit-identical) | §1.4 |

---

## 11. Not verified / note-vs-code mismatches

- No standalone verification of `native_exp2(-inf) == 0` (§1.2).
- Not confirmed whether the output 0 for fully-masked rows matches the reference (whether it gives NaN).
- The measured ns in the int8-perf note are B580 (with clock 2800MHz noted), not re-measured on B70. Static ISA numbers are not time.
- "bias trick = 69% mov reduction" is limited to the microbenchmark K dequant block; the effect on the whole real kernel is −3.8%(K) / −9.5%(K+V) (B580).
- Some env vars in the notes (`SDPA_OCL_BLOCK_SKIP`, `_DKS_ACTIVE`, `_PA_CUR_*`, `_V_I8_PAIRED`, etc.) were removed in the 2026-09 refactor. Read the entries of the §9.3 table above only as a conceptual classification.
- The unparenthesized `LOWER_RIGHT_SHIFT` macro is safe at its current use sites but is a latent risk.
- T9 (bf16 PA guard, plain int4 guard) is noted as "analyzed/not started"; whether it has been applied to the current code was not checked.
