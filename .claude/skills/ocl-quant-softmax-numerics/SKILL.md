---
name: ocl-quant-softmax-numerics
description: Playbook for numerics in Intel Xe DPAS OpenCL attention kernels - online softmax (exp2, alpha, -inf/NaN guards), causal/window/bidir/sink masks, int8/u4 to f16 dequant via the 0x6480 XOR trick, scale/zp factoring, bf16, accuracy verification. Use for sdpa_ocl, KV-cache quantization, WWB/accuracy regressions.
---

# OpenCL attention kernel numerics design / verification playbook

In-depth explanations and evidence (path:line, measurements, hardware) are in `src/plugins/intel_gpu/docs/ocl_perf_guide/03-numerics-softmax-quantization.md`.
DPAS operand roles are in `01-dpas-and-tiling.md`, ISA/spill analysis in `04-spill-isa-profiling.md`, methodology in `05-methodology-and-pitfalls.md`.
Repository rules (AGENTS.md): numerical correctness > performance, no performance claims without measurement, the user does builds/runs (provide command blocks only), always check NaN/Inf/dynamic shape/accumulator/quantization.

## 1. Classification before starting work (answer first)

1. Is this change **bit-preserving** or **order-changing** (changes accumulation/tile order)? For the latter, moving an end-to-end metric (WWB) is not evidence of a bug. Skipping work that adds exactly 0 (`CAUSAL_TILE_SKIP`, prefetches, exact widens) is bit-preserving -> prove it bitwise; a permuted contraction axis (u4, f16 d-major read) or a moved tile split (`PA_CUR_KV_F16`) is order-changing.
2. In which operand is the head dim the lane (prefill KQ: A=K,B=Q / decode KQ: A=Q,B=K)? **Re-derive** the dequant identities **from your own mapping** (do not copy from another kernel).
3. Data path: which of plain SDPA (integer zp i8) / PA cache (non-integer zp, the writer stores 1/scale) / u4 / bf16 is it?

## 2. Online softmax checklist

- [ ] acc is f32 (DPAS `float8`, max/sum f32). Only P is stored to SLM as f16/bf16; the denominator is the f32 sum before rounding.
- [ ] Fold `scale' = scale * LOG2E` once and use `exp2(S*scale' - m*scale')`. Keep S and max in the raw domain → apply masks/sinks with `* iscale` (= 1/scale).
- [ ] `ok = isfinite(m_new)`; if `!ok`, exp_tile=0, alpha=1, keep the max state (prevents `-inf - -inf = NaN`).
- [ ] epilogue: `inv_l = (l > 0) ? recip(l) : 0` (fully-masked row = 0, not NaN).
- [ ] If you change the loop bounds (causal_k, window_k0_begin), change `first`/`last` too. Round the window lower bound **down** to the k0 tile boundary.
- [ ] A private array with a runtime index (alpha[]) goes to scratch → compile-time select chain.
- [ ] `ok ? exp2(..) : 0` per block becomes an IGC goto/join that blocks interleaving → compute exp2, then `select(0, e, ok ? -1 : 0)` (bit-identical; the discarded exp2 of `S + inf` never escapes). +2.6% when reverted (chapter 03 §1.4).
- [ ] decode: scale is pre-multiplied into the scores (Q pre-scaled), the mask value is `VAL_MIN` (not −INF: when the whole partition is masked, `MIN-MIN=0`), `exp_sums/max_logits` are relative to the partition's own max. The partition count/direct-output branch must be exactly complementary to `effective_seq_len` in finalization.

## 3. Mask checklist

- [ ] Do the causal upper bound `causal_k` + block skip (`causal_block_clear`) + per-element predicate use the **same offset** (lower-right: `max(0,k-q)`, PA: `past_len`)?
- [ ] For a dynamic mask, `MASK_KIND` is a host guess → is there a runtime `MSK_D2==1`/`MSK_D3==1` clamp (without it, OOB: `CL_OUT_OF_RESOURCES` on Xe2)?
- [ ] bidir (`token_type_ids`): union `(causal∩window) ∪ own group`, only on the query side `[gb,ge)`. Coordinates: `token_type_ids` = LOCAL, everything else = KEY (`+query_position_offset`). Never access an empty tensor (`count==0`).
- [ ] sink: prefill seeds the online state (`S_max=sink_raw`, `S_max_tile=sink_raw*scale`, `S_sum` is 1 in only one place, `sg_i_kq==0`). decode only on partition 0. Watch the domain difference (prefill raw / decode pre-scaled). Use sink values in tests on the order of `log(kv_len)`.
- [ ] Unwritten cache slots: since `NaN + -inf = NaN`, clamp/0 them at load. Also watch `0*NaN` in V (force both scale and zp to 0).
- [ ] The jit macros / kernel signature / host arg push must match 3-way (precedent: arg shift incidents with HAS_SINK_INPUT, HAS_QQ_BIAS).
- [ ] Last-tile-only mask (`CAUSAL_MASK_LAST_ONLY`): valid only when `wgTK % wgTQ == 0` and no bidir / attn mask / MIXED / lower-right; with a window, an earlier tile still needs the mask while `k0 + W <= wg_j0 + wgTQ - 1` (chapter 03 §2.7).
- [ ] A window **no prompt reaches** (phi-3.5/phi-4: 262144 from HF config) still counts as SWA: `MICRO_MATH` is excluded and a `W == 0` gate drops fast paths → gate on what the window cuts.
- [ ] Mask tests must be **uncompressed** rows (compressed tolerance 25e-3 / u4 75e-3 PASSes a dropped window term) + a negative control that removes the term.

## 4. int8/u4 → f16 widen checklist

Formula: `as_half(0x6480 ^ (ushort)(byte & 0xFF)) == (float)s + 1152` (s=signed int8).
Derivation: `0x6400` = 2^10(E=25), ulp=1 in [1024,2048); `0x6480` = 1024+128; `u ^ 0x80` = s+128 (sign→offset binary); combined, 1024+s+128.
E.g.: s=−3 → 0xFD ^ 0x6480 = 0x647D → mantissa 125 → 1149 = 1152−3. s=−128 → 0x6400 = 1024, s=127 → 0x64FF = 1279. u4: `as_half(0x6400 | n) == 1024 + n`. u8: `0x6400 | u`.

- [ ] Extract bytes with shift+mask (`as_char4` produces a `:b` deinterleave). 4 bytes → 2 VNNI dwords with dword arithmetic (`... ^ 0x64806480u`).
- [ ] **Fold the bias into zp**: `(wide - (zp+1152)) * scale` (a separate bias subtraction blows up the adds).
- [ ] **Exact only for integer zp** (f16 ulp@1152 = 1.0). Forbidden for non-integer zp in PA cache/u4/int4 → explicit half `(q - zp)*scale`, or, as in decode, remove the bias in the **float score correction**.
- [ ] With a non-integer zp the **widen alone** can still skip the convert: `I8_TO_HALF(w16,sh) = as_half(((w16>>sh)&0xFF) ^ 0x6480) - 1152` / `U4_TO_HALF` on a 16-bit word of the read, `- 1152` kept as its own (exact) op, then `(q - zp)*sc`. Bit-identical to `(half)(char)b`; the convert chain (byte deinterleave + strided convert + repack, 4-5 movs/elem) disappears (i8 MIXED loop 7,690 -> 5,842 inst, -11%; chapter 03 §4.8).
- [ ] Compute the term that undoes the bias (`k_corr`) with the **f16-rounded** `Q*sc` that DPAS saw (doing it in float gives 2.3x the error, which the 6e-3 threshold test does not catch).
- [ ] bf16: no trick (the ulp=1 range is only [128,256)). float dequant → bf16 encode. Use `as_char((uchar)b)` (`(char)(uchar)b` saturates to 127). Keep scale/zp tensors as f16. Read the runtime scale dtype from the layout, `SCALE_TO_FLOAT` (bf16 = `as_float((uint)x << 16)`).
- [ ] u4: K adjacent packing (forced by the writer WG split), V split packing (keeps lane==dim). For K in sdpa_ocl MIXED, the depth-axis permute `win+2L+par` is paid only once, in Q staging. Nibble selection is lane-uniform, so it folds into the shift amount.

## 5. dequant identities (where to factor)

| Term | Where it folds |
|---|---|
| BY_TOKEN K sc/zp | score post-processing per-lane (`sc*(S - (zp+bias)*sum_d Q)`), `sum_d Q` reduced 1x per head |
| BY_CHANNEL K sc | multiplied into Q (A operand) (lane==head dim) |
| BY_CHANNEL K zp | subtract one constant `k_corr` from the scores (per page·head) |
| V sc | per-lane multiply into P (A operand). **The denominator stays sum(P)** |
| V zp | subtracted in the V B operand (it varies along the key axis, so it is the only broadcast; if constant per lane, it folds into the source region for free) |

Hoist per-key scale/zp out of the depth/key loops (inside the loop it is 128~256 SIMD-1 loads per k0).

## 6. "exact" MIXED (compressed cache + current chunk)

Cache `[0,past_len)` = quantized dequant, current `[past_len,k)` = raw f16 Kc/Vc. When a k0 tile crosses `past_len`, cut the tile with `k_chunk` so one iteration reads only one source; rows after `k_chunk` are masked, `k0 += k_chunk`. Splitting at page granularity (GRAN=0) gives wrong results.
An **f16** cache holds the new rows bit for bit, so it reads them from the page (`PA_CUR_KV_F16` only for compressed caches since 2026-10-27): the Kc branch cost 5-14% per call.

## 7. Verification playbook (the user runs things — provide command blocks, `--device_suffix=1` mandatory)

1. **Suspect the test data first**: with N(0,0.1), softmax is nearly uniform → Q/K misreads hide. Add sharp copy cases with `logit_scale_gain=128` (constant scale) / `runtime_scale_multiplier` (Xe2+).
   LCG: in `InputGenerateData(start, range, res, seed)`, if `range*res` is a power of 2 and ≤ the row length, every row is identical → use res such as 31/37/1000, and **replay the LCG in Python** first to check the logit spread and the bug's effect. Never use the same seed for Q/K/V.
2. **Run the negative control on the old code first** and compare prediction to observation (a new test is meaningful only if it FAILs on the buggy code).
3. **layer-0 identical-input oracle**: in a chained network, compare only at the first layer. Compare against the output dtype ULP (f16 ≈ 4.9e-4 relative): max ≈ 1 ULP means correct. Later layers have chaotic amplification (1600x, plateau), so correct and buggy cannot be told apart. Self-check whether a dump comparison is even possible by checking that the causal-isolated region (tokens `[0,tile)`) is bit-identical.
4. **WWB similarity is not a kernel gate** (small models are sensitive to ±0.05; minicpm4-0.5b −0.070 was judged not a bug).
5. Classify bit-preserving/order-changing before interpreting toggle results. Never attribute from a one-sided control (e.g. the head-512 failure was not sdpa_ocl but the `dynamic_quantize` 256 cap — run the `TEST_USE_SDPA_OCL=0` control first).
6. **Confirm the kernel that actually ran** (cliloader / `get_kernels_dump_info()`): if the gate rejects, it silently falls through to another backend that is correct, so green is not proof. set-equality (FAIL→PASS set = gate-accepted set).
7. Pre-verify offset/nibble/page agreement in Python (`test/check_by_channel_offsets.py`, `test/check_u4_offsets.py`, `test/check_u4_page_read.py`; including negative controls).
8. Static ISA (instCount/mov count) is not time. Performance is measured by the user (cliloader per-kernel ns), labeled B580/B70/DG2 separately, measurements vs hypotheses distinguished.

## 8. Common traps, one-line summary

`-inf-(-inf)=NaN` / whole-partition mask uses VAL_MIN / NaN in unwritten slots is not erased by the mask / never fold +1152 for non-integer zp / k_corr f16 rounding / `(char)(uchar)` saturate /
sink double counting / LOCAL vs KEY coordinates / dynamic mask kind OOB / bf16 scale misread / dynamic_quantize 256 cap / i8 BY_CHANNEL requantize ignores in_data_pitch (unfixed, `pa_kv_cache_update_ref.cl:297`).
Full table (23 items): chapter 03 §10.
