# KV Cache Quantization in OpenVINO CPU Plugin

Design document for the KV cache codec system: TurboQuant (TBQ), PolarQuant, u8
quantization, and the unified `mha_turboq` attention kernel.

## Overview

The system compresses KV cache entries during LLM decode to reduce memory
bandwidth and footprint. Three codec families are supported, independently
configurable for K and V caches:

| Codec             | Bits/element | Record bytes (dim=128) | Method                                   |
|-------------------|--------------|------------------------|------------------------------------------|
| **f32** (none)    | 32           | 512                    | No compression                           |
| **u8** (GS=128)   | 8 + scale/zp | 136                    | Uniform min-max quantization             |
| **u8** (GS=32)    | 8 + scale/zp | 160                    | Uniform min-max, 4 groups                |
| **u4** (GS=128)   | 4 + scale/zp | 72                     | Uniform min-max, nibble-packed           |
| **u4** (GS=32)    | 4 + scale/zp | 96                     | Uniform min-max, 4 groups                |
| **u8 by-channel** | 8 + scale/zp | 128 + 8/channel        | Per-channel scale/zp across token groups |
| **TBQ4**          | 4.25         | 68                     | WHT rotation + Lloyd-Max codebook        |
| **TBQ3**          | 3.25         | 52                     | WHT rotation + Lloyd-Max codebook        |
| **TBQ4+QJL**      | 4.25 + signs | 88                     | TBQ3 + 1-bit QJL sign correction         |
| **TBQ3+QJL**      | 3.25 + signs | 72                     | TBQ2 + 1-bit QJL sign correction         |
| **Polar4**        | ~4           | ~68                    | Polar coordinate tree decomposition      |
| **Polar3**        | ~3           | ~52                    | Polar coordinate tree decomposition      |

---

## Generic Pipeline

Every decode step executes four phases. K and V can use different codecs
independently.

```
                            KV Cache (packed)
                            +------------------+
  New K/V token (f32)       |  K: [B, Hk, N, packed_K_bytes]  |
         |                  |  V: [B, Hk, N, packed_V_bytes]  |
         v                  +------------------+
  +---------------+                |
  | compress_cache|  <-- Write Path (once per new token)
  | (quantize)    |                |
  +---------------+                |
         |                         |
         v                         v
  Store in cache           +====================+
                           |   mha_turboq()     |  <-- Read Path (every decode step)
                           |                    |
  Q (f32/f16/bf16) ------> | Phase 0: Prepare Q |
                           |   rotate + project |
                           |                    |
                           | Phase 1: Q . K     |
                           |   score = dot(Q_rot, dequant(K_packed))
                           |   per-token codec dispatch
                           |                    |
                           | Phase 2: Softmax   |
                           |   mask + normalize |
                           |                    |
                           | Phase 3: V Accum   |
                           |   acc += w * dequant(V_packed)
                           |   per-token codec dispatch
                           |                    |
                           | Phase 4: Reduce    |
                           |   cross-thread sum |
                           |   + QJL correction |
                           |   + inverse rotate |
                           +====================+
                                    |
                                    v
                              Output (f32)
```

---

## Write Path: compress_cache()

Called once per new token to quantize and store K/V into the cache.

```
compress_cache(cur, dst, L0, ...)
    |
    +-- is_codec?
    |     +-- is_polar? --> polarq_quantize_single()
    |     |                     polarq_quantize_head() per (B, H, L)
    |     +-- else -------> turboq_quantize_single()
    |                           turboq_quantize_head() or
    |                           turboq_quantize_head_qjl() per (B, H, L)
    +-- is_u8? ----------> u8_quantize_single()
    |                           attn_quant_u8() per (B, H, L, group)
    +-- else (raw) ------> attn_memcpy_single()
                                memcpy f32/f16/bf16
```

### TurboQuant Quantize (per head, dim=128)

```
Input: x[128] (f32)
  |
  v
(1) norm = ||x||                          Cost: 128 FMAs
  |
  v
(2) unit = x / norm                       Cost: 128 divs
  |
  v
(3) rotated = WHT(signs, unit) * sqrt(d)  Cost: O(d log d) = 896 ops
  |                                        [butterfly passes + sign flips]
  v
(4) indices[i] = quantize(rotated[i])     Cost: 128 * n_boundaries comparisons
  |               boundary scan             [branchless linear scan]
  v
(5) pack indices into bytes               Cost: 128 * bit_ops
  |   4-bit: 2 per byte (64 bytes)
  |   3-bit: 8->3 bytes groups (48 bytes)
  v
(6) store norm (fp32, 4 bytes)            Optional: norm correction
  |
  v
Output: [packed_bytes | norm_fp32]
         64+4=68 bytes (TBQ4)
         48+4=52 bytes (TBQ3)
```

### TurboQuant+QJL Quantize (per head)

```
Steps (1)-(5) same as TBQ, but with (bits-1) Lloyd-Max bits.
Then:
  |
  v
(6) residual = rotated - codebook[indices]  Cost: 128 subs
  |
  v
(7) gamma = ||residual||                    Cost: 128 FMAs
  |
  v
(8) projected = QJL_project(residual)       Cost: O(d log d) WHT
  |              [independent WHT signs]     or O(d^2) dense
  v
(9) sign_bits = sign(projected) -> pack     Cost: 128 comparisons + pack
  |              16 bytes (128 bits)
  v
Output: [packed_indices | sign_bytes(16) | gamma_fp32 | norm_fp32]
         48+16+4+4 = 72 bytes (TBQ3+QJL)
         32+16+4+4 = 56 bytes (TBQ2+QJL, used in TBQ3_QJL)
```

### u8 Quantize (per group of group_size elements)

```
Input: x[group_size] (f32)
  |
  v
(1) min_val = min(x), max_val = max(x)
  |
  v
(2) scale = (max - min) / 255
    zp = -min / scale
  |
  v
(3) quantized[i] = round(x[i] / scale + zp)   clamp to [0, 255]
  |
  v
Output: [u8_data(group_size) | scale_fp32 | zp_fp32]
```

---

## Asymmetric Codec: Rotation Flow

When K and V use different codecs, rotation is applied selectively per side.
The table below shows what happens at each phase for four representative
configurations. R = forward rotation (WHT or dense), R^T = inverse rotation.

```
+-------+-------+------------------+------------------+------------------+------------------+
| K     | V     | Write K          | Write V          | Read (Q,K,V)     | Output fixup     |
+=======+=======+==================+==================+==================+==================+
| tbq   | tbq   | R(k), quantize,  | R(v), quantize,  | Q: R(q)          | R^T(accum)       |
|       |       | pack + norm      | pack + norm      | K: dot(R(q),     |   restores V to  |
|       |       |                  |                  |    packed R(k))  |   original domain|
|       |       |                  |                  | V: accum in      |                  |
|       |       |                  |                  |    rotated domain|                  |
+-------+-------+------------------+------------------+------------------+------------------+
| tbq   | u8    | R(k), quantize,  | u8 quantize      | Q: R(q)          | no inv. rotation |
|       |       | pack + norm      | (scale+zp, no R) | K: dot(R(q),     |   V already in   |
|       |       |                  |                  |    packed R(k))  |   original domain|
|       |       |                  |                  | V: u8 dequant,   |                  |
|       |       |                  |                  |    accum plain   |                  |
+-------+-------+------------------+------------------+------------------+------------------+
| u8    | tbq   | u8 quantize      | R(v), quantize,  | Q: NOT rotated   | R^T(accum)       |
|       |       | (scale+zp, no R) | pack + norm      | K: dot(q,        |   restores V to  |
|       |       |                  |                  |    dequant(u8 k))|   original domain|
|       |       |                  |                  | V: accum in      |                  |
|       |       |                  |                  |    rotated domain|                  |
+-------+-------+------------------+------------------+------------------+------------------+
| u8    | u8    | u8 quantize      | u8 quantize      | Q: NOT rotated   | no inv. rotation |
|       |       | (scale+zp, no R) | (scale+zp, no R) | K: dot(q,        |   V already in   |
|       |       |                  |                  |    dequant(u8 k))|   original domain|
|       |       |                  |                  | V: u8 dequant,   |                  |
|       |       |                  |                  |    accum plain   |                  |
+-------+-------+------------------+------------------+------------------+------------------+

Key insight: Q rotation is driven by K's codec, not V's.
  - K has codec  -->  Q is rotated (Q-prod: R(q).R(k) = q.k)
  - K is plain   -->  Q stays plain (direct q.k dot product)

V's codec drives the output fixup:
  - V has codec  -->  accum is in rotated domain, needs R^T at the end
  - V is plain   -->  accum is in original domain, skip R^T

This is correct because rotation is orthogonal: R^T(sum(w * R(v))) = sum(w * v).
```

---

## Read Path: mha_turboq()

Unified multi-head attention kernel supporting all codec combinations.

### Phase 0: Prepare Q

```
turboq_prepare_query(q_input, out_q, rotate, qjl, ...)
  |
  +-- Convert Q to f32 (from f16/bf16 if needed)
  |
  +-- rotate? --> turboq_rotate_forward(q, dst)
  |                 WHT mode: dst = H * diag(signs) * q / sqrt(d)
  |                 Dense mode: dst = Q_dense * q
  |
  +-- qjl? -----> turboq_qjl_project_forward(dst, dst+S)
                    WHT mode: projected = H * diag(qjl_signs) * q_rot / sqrt(d)
                    Dense mode: projected = S * q_rot

Output layout:
  Standard: [q_rotated(S)]
  QJL:      [q_rotated(S) | q_projected(S)]
```

### Phase 1: Q.K Scores

```
turboq_foreach_kv(B, Hk, ...) parallel over [B, Hk, kv_len]
  |
  dispatch_q_precision → dispatch_codec → score_tokens(make_score(codec))
  |
  For each cached K token, record_qk_dot() dispatches via if constexpr:
  |
  +-- is_qjl? -----> turboq_codec_qk_dot_qjl():
  |                     base = codec_dot(indices, q_rot, TBQCodecN) * norm_scale
  |                     correction = codec_dot(signs, q_projected, SignCodec)
  |                                  * norm * inv_sqrt_dim * sqrt(pi/2)/d * gamma
  |                     score = base + correction
  |
  +-- is_polar? ----> polar_token_qk_dot():
  |                     polar_precompute_token_simd()  [decompress tree]
  |                     polar_qk_dot_precomputed_simd()  [interleaved dot]
  |
  +-- grouped? -----> Group loop (U8/U4):
  |                     for each group: codec_dot(data, q, group_size, GroupedCodec)
  |
  +-- default ------> codec_dot(data, q, hd, InnerCodec) * scale  (TBQ/Raw)
```

**TBQ Codec Dot Product Detail (AVX-512 SIMD):**

```
For each group of W=16 elements:
  +-- Load W packed nibbles from K record
  |     4-bit: load 8 bytes, split lo/hi nibbles
  |     3-bit: load 6 bytes, shift+mask to 8 indices
  |
  +-- Codebook lookup: vpermps(codebook_16, indices)
  |     Single instruction: 16 lookups in parallel
  |
  +-- FMA: dot += codebook_vals * q_rotated[j]
  |
  Final: score = reduce(dot) * norm * inv_sqrt_dim
```

### Phase 2: Softmax

```
turboq_softmax()
  |
  attn_softmax_kernel<float>(scores, mask, alibi, sink, ...)
    +-- Apply attention mask (causal)
    +-- Apply alibi positional bias (if present)
    +-- Handle sink tokens
    +-- Numerically stable softmax: exp(x - max) / sum
```

### Phase 3: V Accumulation

```
turboq_foreach_kv(B, Hk, ...) parallel over [B, Hk, kv_len]
  |
  Per-thread init: memset accumulators to 0
  |
  dispatch_codec → accum_tokens(make_v_accum(codec))
  |
  For each cached V token, record_v_accum() dispatches via if constexpr:
  |
  +-- is_qjl? -----> Two codec_weighted_accum passes:
  |                     Pass 1: codec_weighted_accum(indices, TBQCodecN, norm_scale) → accum
  |                     Pass 2: codec_weighted_accum(signs, SignCodec, sign_scale) → sign_accum
  |                     (sign_accum stored at accum + dim)
  |
  +-- is_polar? ----> polar_token_v_accum():
  |                     polar_precompute_token_simd()  [decompress tree → cartesian]
  |                     codec_weighted_accum(cartesian, RawCodec<float>) → accum
  |
  +-- grouped? -----> Group loop (U8/U4):
  |                     for each group: codec_weighted_accum(data, GroupedCodec) → accum
  |
  +-- default ------> codec_weighted_accum(data, InnerCodec, scale) → accum  (TBQ/Raw)

Accumulators are in the ROTATED domain (no inverse rotation yet).
```

### Phase 4: Reduce + Inverse Rotation

```
turboq_reduce() --> turboq_reduce_head() per (B, H)
  |
  Step 1: Sum per-thread accumulators (SIMD)
  |   dst[j] = sum_t(accum_t[j])
  |
  Step 2 (QJL only): Sign correction
  |   sign_reduced[j] = sum_t(sign_accum_t[j])
  |   correction = QJL_project_inverse(sign_reduced)
  |   dst[j] += sqrt(pi/2) / dim * correction[j]
  |
  Step 3: Inverse rotation
      WHT:   output = diag(signs) * H * dst / sqrt(d)
      Dense: output = Q^T * dst
      None:  output = dst  (when v_rotation_fused)
```

---

## Record Layouts (dim=128)

### TBQ4 (68 bytes)
```
+----------------------------------+----------+
| packed 4-bit indices (64 bytes)  | norm(4B) |
| 2 indices per byte, lo nibble 1st|  fp32    |
+----------------------------------+----------+
```

### TBQ3 (52 bytes)
```
+----------------------------------+----------+
| packed 3-bit indices (48 bytes)  | norm(4B) |
| 8 indices -> 3 bytes per group   |  fp32    |
+----------------------------------+----------+
```

### TBQ4+QJL (88 bytes)
```
+-------------------+-------------+----------+----------+
| 3-bit indices     | sign bits   | gamma(4B)| norm(4B) |
| (48 bytes)        | (16 bytes)  | fp32     | fp32     |
+-------------------+-------------+----------+----------+
```

### TBQ3+QJL (72 bytes)
```
+-------------------+-------------+----------+----------+
| 2-bit indices     | sign bits   | gamma(4B)| norm(4B) |
| (32 bytes)        | (16 bytes)  | fp32     | fp32     |
+-------------------+-------------+----------+----------+
```

Note: norm is always the last 4 bytes of every TBQ/QJL record, enabling
a single `turboq_norm_scale(data, record_bytes, dim)` function for all layouts.

### Polar4/Polar3 (variable, ~68/52 bytes)
```
+----------------------------------------------+----------+
| mixed-width packed angles per tree level     | norm(4B) |
| (bits-per-level varies by level and config)  | fp32     |
+----------------------------------------------+----------+
```

### u8 (128 bytes + 8/group)
```
+----------------------------+  +-------------------------+
| u8 values (128 bytes)      |  | scale_zp per group:     |
| 1 byte per element         |  | [scale_f32 | zp_f32] x  |
+----------------------------+  | (128/group_size) groups  |
                                +-------------------------+
```

### u4 (64 bytes + 8/group)
```
+----------------------------+  +-------------------------+
| u4 values (64 bytes)       |  | scale_zp per group:     |
| 2 elements per byte        |  | [scale_f32 | zp_f32] x  |
| high nibble = even elem    |  | (128/group_size) groups  |
| low nibble  = odd elem     |  +-------------------------+
+----------------------------+
```
Note: u4 nibble order differs from TBQ4 (which uses low nibble = even element).
`Simd::load_cvt_u4()` handles the u4 convention; `unpack_4bit_nibbles()` handles TBQ4.

---

## Unified Codec Infrastructure

Two layers of codec abstraction: **element codecs** (SIMD decode of W elements)
and **record codecs** (per-record orchestration of groups, scale, and dispatch).

### Element Codecs

All element codecs provide the same interface for `codec_dot` / `codec_weighted_accum`:

```cpp
struct SomeCodec {
    Simd decode(const uint8_t* base, int bit_offset) const;  // decode W elements
    static constexpr int group_bits;      // bits consumed per decode
};
```

**Generic SIMD templates** (in `attn_quant_turboq.cpp`):

| Template | Purpose |
|----------|---------|
| `codec_dot<QT, Codec>` | SIMD dot product: `sum(codec.decode(data) * q)`, 4x unrolled + tail |
| `codec_weighted_accum<Codec>` | SIMD V accumulation: `accum += weight * outer_scale * codec.decode(data)`, multi-head, 4x unrolled |

**Element codec types**:

| Codec | Defined in | `decode()` returns |
|-------|-----------|-------------------|
| `TurboQCodec4/3/2` | `turboq_codecs.hpp` | `codebook[idx]` via vpermps (constructed via default ctor) |
| `PolarCodec5/4/3/2` | `polar_codecs.hpp` | Polar centroid decode (constructed from `PolarqLevelLUT`) |
| `U8Codec` | `codecs.hpp` | `(load_cvt(byte) - zp) * scale` |
| `U4Codec` | `codecs.hpp` | `(load_cvt_u4 - zp) * scale` (ISA-agnostic) |
| `RawCodec<KT>` | `codecs.hpp` | `Simd::load_cvt(KT*)` — identity decode for f32/f16/bf16 |
| `SignCodec` | `attn_quant_turboq.cpp` | Expand packed 1-bit signs to ±1.0 (for QJL) |

### Record Codecs

Record codecs orchestrate element codecs for a full cache record — handling
per-record scale, group iteration, and special scoring formulas.

| Record Codec | Purpose |
|-------------|---------|
| `RecordCodec<InnerCodec>` | Single element codec + per-record scale. TBQ: scale = norm/√d. Raw: scale = 1.0. |
| `GroupedCodec<InnerCodec>` | Per-token view with resolved szp pointer, iterates groups. For U8/U4. |
| `GroupedRecordCodec<InnerCodec>` | Factory holding PlainTensor. `for_token()` → `GroupedCodec`. |
| `U8ByChannelGroupedCodec` | Per-token view for by-channel u8. `group_codec()` loads vector scale/zp per Simd::width chunk. |
| `ByChannelRecordCodec` | Factory holding scale_zp tensor. `for_token()` → `U8ByChannelGroupedCodec`. |
| `QJLRecordCodec<InnerCodec>` | Base codebook + SignCodec correction. Two codec_dot/codec_weighted_accum passes. |
| `PolarRecordCodec` | Polar tree decomposition. QK: custom interleaved dot. V: decompress + RawCodec accum. |

**Detection traits** (SFINAE, in `codecs.hpp`) — dispatch on interface, not tags:

| Trait | Detects | Meaning |
|-------|---------|---------|
| `is_head_grouped_v<T>` | `group_codec()` method | head_dim split into sub-groups |
| `is_token_indexed_v<T>` | `for_token()` method | scale/zp in external tensor indexed by token |
| `is_polar_v<T>` | `bpl` member | polar tree decomposition |
| `is_qjl_v<T>` | `index_bytes` member | QJL sign correction |

**Runtime dispatch**: `dispatch_codec(CacheCodec, hd, phase_bits, group_size, scale_zp, fn)`
maps the runtime `CacheCodec` enum to the concrete record codec type, calling `fn(codec)`.

### Scoring/Accumulation Flow

```
dispatch_q_precision → dispatch_codec → score_tokens/accum_tokens
    → record_qk_dot / record_v_accum
        → codec_dot / codec_weighted_accum (element codec inner loop)
```

**Norm scaling**: `turboq_norm_scale(data, record_bytes, dim)` reads the per-token
norm (always the last 4 bytes of a TBQ/QJL/Polar record) and returns `norm / sqrt(dim)`.
Works for all record layouts because norm is stored at a consistent position.

---

## TurboQuant Mathematical Foundation

### Rotation: Why and How

KV cache vectors have non-uniform energy distribution across dimensions.
Rotation spreads energy uniformly, making scalar quantization near-optimal.

**Walsh-Hadamard Transform (WHT):**
```
R(x) = H * diag(signs) * x / sqrt(d)
```
- `H` = Hadamard matrix (butterfly structure, O(d log d))
- `signs` = random +-1 vector (cached, deterministic seed)
- Self-inverse: `R^{-1}(y) = diag(signs) * H * y / sqrt(d)`

**Q-prod optimization:** Since `<q, k> = <R(q), R(k)>` (orthogonal),
we rotate Q once instead of inverse-rotating all N cached K tokens.

### Lloyd-Max Codebook

Centroids minimize MSE for the Beta distribution of rotated unit-norm
coordinates. Quantization = branchless scan over decision boundaries
(midpoints between adjacent centroids).

### QJL Sign Correction

For the residual `r = rotated - codebook[idx]`:
```
<q, r> ~= gamma * sqrt(pi/2) / d * <S(q), sign(S(r))>
```
- `S` = independent WHT projection (different signs from MSE rotation)
- `gamma = ||r||` stored per record
- 1-bit per dimension (16 bytes for dim=128)

---

## Configuration

Independent K/V configuration via OpenVINO properties:

```json
{
    "KEY_CACHE_CODEC": "tbq4",
    "VALUE_CACHE_CODEC": "polar4",
    "ATTENTION_BACKEND": "SDPA",
    "KEY_CACHE_GROUP_SIZE": 32,
    "VALUE_CACHE_GROUP_SIZE": 32
}
```

Supported codec values: `tbq4`, `tbq3`, `tbq4_qjl`, `tbq3_qjl`, `polar4`, `polar3`.
Supported precision values: `u8`, `u4`, `f32` (+ `f16`, `bf16` on supported hardware).

### Environment Variables

| Variable | Effect |
|----------|--------|
| `OV_TURBOQ_NORM_CORRECTION` | Store corrected norm so dequant norm matches original |
| `OV_TURBOQ_FUSED_QUANTIZE` | Use SIMD fused quantize path |
| `OV_TURBOQ_SKIP_ROTATION` | Skip Q rotation (debug only, breaks correctness) |

---

## Key Source Files

| File | Purpose |
|------|---------|
| `scaled_attn.cpp` | SDPA node: write path, cache management, mha dispatch |
| `codecs.hpp` | All codecs: inner (U8/U4/Raw), record (Grouped/ByChannel/Polar/QJL/TBQ), detection traits, CacheCodec enum, select_codec |
| `attn_quant_turboq.cpp` | SignCodec, codec_dot, codec_weighted_accum, record_qk_dot/record_v_accum, mha_turboq kernel |
| `attn_quant_turboq.hpp` | Public API: quantize, head_bytes, mha_turboq |
| `turboq_rotation.hpp` | WHT, dense rotation, QJL projection dispatch |
| `turboq_rotation.cpp` | Matrix/sign generation, caching, mode control |
| `turboq_tables.h` | Lloyd-Max codebook centroids and boundaries |
| `turboq_codecs.hpp` | TBQ element codec structs (vpermps codebook decode, constructed via default ctor) |
| `polar_codecs.hpp` | Polar element codec structs (tree decomposition, constructed from LUT) |
| `polarq_tables.h` | Polar quantization centroid tables |
| `simd.hpp` | ISA-agnostic SIMD abstraction (AVX-512/AVX2/scalar), broadcast ctor, load_cvt_u4 |
