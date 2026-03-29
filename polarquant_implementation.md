# PolarQuant Implementation Plan

## Background

The current TurboQuant (TBQ) KV cache codec applies a random orthogonal rotation
then independently scalar-quantizes each coordinate with Lloyd-Max codebooks.
This is **not** what the PolarQuant paper describes.

PolarQuant (arxiv 2502.02617) instead performs a **recursive polar decomposition**
after rotation, quantizing angles at each level with level-specific codebooks.
Higher levels have increasingly concentrated angle distributions, requiring fewer
bits — enabling a non-uniform bit allocation that is more bit-efficient.

TurboQuant (arxiv 2504.19874) combines PolarQuant (base codec) + QJL (1-bit
residual correction). Our diagnostic tests showed QJL adds more variance than it
removes, but those tests used rotated-scalar-Lloyd-Max as the base — not actual
PolarQuant. This implementation adds true PolarQuant to evaluate the combination
properly and to measure whether the polar base codec itself is superior.

## Algorithm: PolarQuant Quantization

**Scope constraint**: `head_dim` must be a power of two and divisible by 64 (same
as current TBQ). The tree depth is `L = log2(head_dim)`. Known target dimensions:
- `head_dim=128` (Llama, Qwen, Phi) → L=7 levels, 127 angles
- `head_dim=256` (Gemma 3) → L=8 levels, 255 angles

The algorithm, codebook tables, and bit allocations are parameterized by `head_dim`.
Codebooks depend only on the level (not `head_dim`), since the angle distribution
at level k depends on effective d.f. = 2^(k-1), which is the same regardless of
total dimension. So the same level-k codebook works for any `head_dim >= 2^k`.

Input: float vector `x` of dimension `dim` (power-of-two).

```
L = log2(dim)   // tree depth: 7 for dim=128, 8 for dim=256

1. norm = ||x||
2. unit = x / norm
3. rotated = Q * unit * sqrt(dim)     // same rotation as current TBQ
   // rotated has total L2 norm = sqrt(dim), NOT norm*sqrt(dim).
   // norm is stored separately and applied once during reconstruction.
4. Recursive polar decomposition (L levels):
   Level 1: pair coordinates (rotated[2i], rotated[2i+1])
     angle[1][i] = atan2(rotated[2i+1], rotated[2i])        // in [0, 2*pi)
     radius[1][i] = sqrt(rotated[2i]^2 + rotated[2i+1]^2)
   Level k (k=2..L): pair radii from level k-1
     angle[k][i] = atan2(radius[k-1][2i+1], radius[k-1][2i])  // in [0, pi/2]
     radius[k][i] = sqrt(radius[k-1][2i]^2 + radius[k-1][2i+1]^2)
5. Final single radius[L][0] = sqrt(dim) (sanity check — this is the norm
   of the rotated unit vector scaled by sqrt(dim), NOT the original norm)
6. Quantize each angle[level][i] using level-specific Lloyd-Max codebook
7. Pack: [level1_indices | level2_indices | ... | levelL_index | fp32_norm]
```

### Angle count per level

General formula: level k has `dim / 2^k` angles. Total = `dim - 1` angles + 1 norm.

**dim=128 (L=7):**

| Level | Angles | Domain  |
|-------|--------|---------|
| 1     | 64     | [0, 2π) |
| 2     | 32     | [0, π/2]|
| 3     | 16     | [0, π/2]|
| 4     | 8      | [0, π/2]|
| 5     | 4      | [0, π/2]|
| 6     | 2      | [0, π/2]|
| 7     | 1      | [0, π/2]|
| **Total** | **127** | + 1 norm |

**dim=256 (L=8):**

| Level | Angles | Domain  |
|-------|--------|---------|
| 1     | 128    | [0, 2π) |
| 2     | 64     | [0, π/2]|
| 3     | 32     | [0, π/2]|
| 4     | 16     | [0, π/2]|
| 5     | 8      | [0, π/2]|
| 6     | 4      | [0, π/2]|
| 7     | 2      | [0, π/2]|
| 8     | 1      | [0, π/2]|
| **Total** | **255** | + 1 norm |

### Angle distributions

After random orthogonal rotation, coordinates are near-i.i.d. Gaussian.

- **Level 1**: `atan2(z2, z1)` for i.i.d. Gaussians → **uniform on [0, 2π)**.
- **Level k > 1**: `atan2(r_odd, r_even)` where both radii are chi-distributed.
  Distribution follows a Beta-like shape on [0, π/2], increasingly concentrated
  around π/4 as the effective degrees of freedom double at each level.
  - Level 2: mild concentration (2 d.f.)
  - Level 7 (dim=128): extremely peaked at π/4 (64 d.f.)
  - Level 8 (dim=256): even more peaked (128 d.f.)

This concentration is the core advantage: higher levels need fewer bits.
The codebook at level k is the same regardless of `head_dim`, since the
distribution depends only on `2^(k-1)` effective degrees of freedom.

## Algorithm: PolarQuant Dequantization (Top-Down Radius Tree)

Reconstruction proceeds **top-down**. The tree reconstructs the normalized
rotated shape (total radius = sqrt(dim)). The stored norm is applied **once**
at the very end, after inverse rotation.

```
L = log2(dim)
1. r_top = sqrt(dim)   // root radius of the normalized tree, NOT norm
2. For level = L..2:
     For each pair i at this level:
       r_left[i]  = r_parent[i] * cos(codebook[level][angle_idx[level][i]])
       r_right[i] = r_parent[i] * sin(codebook[level][angle_idx[level][i]])
3. Now have dim/2 level-1 radii
4. For each level-1 pair i:
     cartesian[2i]   = r[1][i] * cos(codebook[1][angle_idx[1][i]])
     cartesian[2i+1] = r[1][i] * sin(codebook[1][angle_idx[1][i]])
5. Apply Q^T inverse rotation: recovered = Q^T * cartesian / sqrt(dim)
6. Scale by norm: output = recovered * norm
```

Steps 5-6 can be fused as: `output = Q^T * cartesian * (norm / sqrt(dim))`
which is `Q^T * cartesian * norm * inv_sqrt_dim` — same scaling pattern as TBQ.

cos/sin values come from **precomputed lookup tables** indexed by quantized
angle index — no trig at runtime on the hot path.

## Fused QK Dot Product

The query is already rotated: `q_rot = Q * q`. The dot product in the rotated
domain is `sum_i q_rot[i] * k_cartesian[i]`.

**Important**: The existing TBQ pipeline writes **raw scores** in Phase 1
(see `turboq_batch_qk_dot` output). The `d_scale = 1/sqrt(head_dim)` is applied
later in Phase 2 (`turboq_softmax`). PolarQuant must follow the same convention
to avoid double-scaling. The kernel returns the raw dot product including the
norm factor but **not** `d_scale`.

**Norm handling**: The tree reconstructs the normalized shape (total radius =
sqrt(dim)). The stored norm is applied as a single scalar factor. The full
dot product is: `<q, k> = (norm / sqrt(dim)) * <q_rot, k_tree>` where `k_tree`
is the tree-reconstructed Cartesian vector. This gives `norm * inv_sqrt_dim`
as the overall scale — same as TBQ's `fused_qk_dot`.

```
L = log2(dim)
1. Read norm from record
2. Reconstruct dim/2 level-1 radii via top-down tree (L-1 levels of cos/sin lookup)
   r_top = sqrt(dim) for the root, then split at each level using cos/sin LUTs
3. For each level-1 pair i:
     partial = r[1][i] * (q_rot[2i] * cos_lut[angle_idx] + q_rot[2i+1] * sin_lut[angle_idx])
4. Return norm * inv_sqrt_dim * sum(partials)
```

The caller (`mha_turboq` / `mha_polar`) applies `d_scale` in the softmax phase,
exactly as current TBQ does.

**Cost estimate** (provisional — mixed-width unpack cost for 5-bit and 3-bit
fields not yet measured; treat as approximate until packing format is benchmarked):

For dim=128: ~126 multiply+lookups for tree traversal + 128 FMAs for the dot
= ~254 operations. For dim=256: ~254 tree ops + 256 FMAs = ~510 operations.
General: ~(dim-2) tree ops + dim FMAs = ~(2*dim - 2) operations.
Comparable to current TBQ (dim lookups + dim FMAs), but the mixed-width unpack
adds overhead that TBQ's uniform-width packing avoids.

**SIMD strategy**: Process each tree level with vectorized `_mm512_permutexvar_ps`
on cos/sin tables (same pattern as current codebook lookup). Level 1 has dim/2 pairs;
for dim=128 that's 4 AVX-512 iterations, for dim=256 it's 8. Higher levels have
fewer elements.

## Fused V Accumulation

V must be fully dequantized to Cartesian for weighted accumulation:

```
1. Reconstruct dim/2 level-1 radii (same tree traversal as QK, r_top = sqrt(dim))
2. cartesian[2i]   = r[1][i] * cos_lut[angle_idx[1][i]]
   cartesian[2i+1] = r[1][i] * sin_lut[angle_idx[1][i]]
3. scale = norm * inv_sqrt_dim   // same norm handling as QK
4. For each head h:
     accum[h][j] += weight[h] * cartesian[j] * scale
```

Same cost as QK dot, substituting store+accumulate for horizontal reduction.

## Bit Allocation

Bit allocations below are worked examples for dim=128. For dim=256, the same
level codebooks apply (level k has the same distribution regardless of dim);
the tree is one level deeper (L=8), with 128 angles at level 1 instead of 64.
Record sizes scale as `dim * bits_per_coord / 8 + 4` bytes.

### Polar4 (4 bits per coordinate average) — dim=128 example (68 bytes)

Target: 128 * 4 = 512 bits for indices + 32 bits for fp32 norm = 544 bits = 68 bytes.

| Level | Angles | Bits/angle | Total bits | Rationale                              |
|-------|--------|------------|------------|----------------------------------------|
| 1     | 64     | 4          | 256        | Uniform distribution, needs resolution |
| 2     | 32     | 5          | 160        | Mild concentration, benefits from bits |
| 3     | 16     | 4          | 64         | Moderate concentration                 |
| 4     | 8      | 3          | 24         | Narrower distribution                  |
| 5     | 4      | 2          | 8          | Concentrated around π/4                |
| 6     | 2      | 0          | 0          | Fix at π/4 (very peaked)               |
| 7     | 1      | 0          | 0          | Fix at π/4                             |
| **Total** | **127** |       | **512**    |                                        |

Norm: 4 bytes (fp32). **Total: 68 bytes** (matches TURBOQ_HEAD_BYTES_TBQ4).

### Polar3 (3 bits per coordinate average) — dim=128 example (52 bytes)

Target: 128 * 3 = 384 bits for indices + 32 bits for norm = 416 bits = 52 bytes.

| Level | Angles | Bits/angle | Total bits | Rationale                    |
|-------|--------|------------|------------|------------------------------|
| 1     | 64     | 3          | 192        | Uniform, highest impact      |
| 2     | 32     | 3          | 96         | Mild concentration           |
| 3     | 16     | 3          | 48         | Moderate concentration       |
| 4     | 8      | 2          | 16         | Narrow, fewer bits suffice   |
| 5     | 4      | 2          | 8          | Concentrated                 |
| 6     | 2      | 2          | 4          | Very concentrated            |
| 7     | 1      | 0          | 0          | Fix at π/4 (extremely peaked)|
| **Total** | **127** |       | **364**    |                              |

20 spare bits (384 - 364). All angles within a level must use the same bit-width
(required for uniform SIMD unpack). Spare bits can only be used by bumping an
entire level's width — e.g., level 4 from 2→3 bits adds 8 bits (8 angles × 1),
or level 6 from 2→3 bits adds 2 bits. Bumping level 4 to 3 bits uses 8 of the
20 spare, leaving 12 for padding. Per-angle variable widths within a level are
explicitly excluded — they would break the SIMD unpack story.

Final allocation should be determined by Phase 0's per-level MSE analysis,
choosing which levels benefit most from an extra bit while keeping widths uniform
within each level.

**Total: 52 bytes** (matches TURBOQ_HEAD_BYTES_TBQ3).

**Note**: Optimal bit allocation should be computed offline using rate-distortion
theory — allocate more bits to levels with higher angle variance. The tables above
are starting points; the Python codebook script should also output optimal allocations.

## Memory Layout

Record sizes are `dim * bits_per_coord / 8 + 4` bytes (fp32 norm at the end).
Layouts below are for dim=128. For dim=256, the same structure applies with
one more level and proportionally more index bytes at each level.

### Polar4 head record — dim=128 (68 bytes)

```
Offset  Bytes  Content
0       32     Level 1: 64 angles × 4 bits = 256 bits
32      20     Level 2: 32 angles × 5 bits = 160 bits
52      8      Level 3: 16 angles × 4 bits = 64 bits
60      3      Level 4: 8 angles × 3 bits = 24 bits
63      1      Level 5: 4 angles × 2 bits = 8 bits
64      4      fp32 norm
Total:  68 bytes
```

Levels 6-7 fixed at π/4 (zero storage). Record is 68 bytes — exceeds a 64-byte
cache line by 4 bytes, so every access crosses a cache-line boundary. Same issue
as current TBQ4 (also 68 bytes). The norm sits in the second cache line.

### Polar3 head record — dim=128 (52 bytes)

Default layout (bumps level 4 to 3 bits, uses 8 of 20 spare bits, 12 bits padding):

```
Offset  Bytes  Content
0       24     Level 1: 64 angles × 3 bits = 192 bits
24      12     Level 2: 32 angles × 3 bits = 96 bits
36      6      Level 3: 16 angles × 3 bits = 48 bits
42      3      Level 4: 8 angles × 3 bits = 24 bits (bumped from 2→3)
45      1      Level 5: 4 angles × 2 bits = 8 bits
46      1      Level 6: 2 angles × 2 bits = 4 bits (padded to 8 bits)
47      1      Padding (4 unused bits, padded to byte boundary)
48      4      fp32 norm
Total:  52 bytes
```

Level 7 is fixed at π/4 (zero storage). All 7 levels are accounted for in
the decoder: levels 1-6 are read from the record, level 7 uses the fixed
centroid (cos(π/4) = sin(π/4) = 1/√2).

This layout is provisional. Phase 0's MSE analysis may shift the extra bits
to a different level (e.g., level 5 from 2→3 instead of level 4). Any such
change must keep bit-widths uniform within each level.

## Pipeline Integration: PolarQuant fits inside mha_turboq

PolarQuant does **not** require a new attention pipeline. The existing `mha_turboq`
4-phase structure is codec-agnostic — PolarQuant only changes what happens inside
the batch kernel lambdas that `turboq_foreach_kv_chunk` calls.

| Phase | Current TBQ | PolarQuant | Change needed |
|-------|-------------|------------|---------------|
| Prepare Q | `turboq_prepare_query`: `Q * q` | Same rotation | None |
| Phase 1: Q·K | Lambda calls `turboq_batch_qk_dot` | Lambda calls `polarq_batch_qk_dot` | Swap lambda body |
| Phase 2: Softmax | `turboq_softmax` | Identical | None |
| Phase 3: V accum | Lambda calls `turboq_batch_v_accum` | Lambda calls `polarq_batch_v_accum` | Swap lambda body |
| Phase 4: Reduce | `turboq_reduce_head` (thread reduce + Q^T) | Identical — accumulators are in rotated Cartesian domain after polar→Cartesian reconstruction | None |

**Why this works**: After PolarQuant reconstructs Cartesian coordinates from the
angle tree (inside the batch kernel), the result is in the same rotated domain as
TBQ's codebook-lookup output. The pipeline above and below the batch kernels —
query rotation, softmax, thread reduction, inverse rotation — is domain-agnostic.

**Implementation**: Add a codec parameter to `mha_turboq` and dispatch in the
lambda bodies:

```cpp
// Phase 1 lambda — only change is the dispatch
[codec](const uint8_t* const* kv, const float* const* q, float* const* scores,
        int n, int nh, int hd, int b) {
    if (is_polar(codec))
        polarq_batch_qk_dot(kv, q, scores, n, nh, hd, b);
    else
        turboq_batch_qk_dot(kv, q, scores, n, nh, hd, b);
}
```

The `turboq_foreach_kv_chunk` template, prefetching, beam reorder, thread
parallelism, and all plumbing remain untouched. The KV stride comes from
`packed_kv.stride(2)` which is set by `memory.cpp` based on `head_bytes` —
as long as `polarq_head_bytes` returns the correct size, the chunking works.

## Reusable Infrastructure

| Component | File | Reuse |
|-----------|------|-------|
| Rotation matrix Q, Q^T | `turboq_rotation.cpp` | 100% — same random orthogonal rotation |
| Query preparation | `attn_quant_turboq.cpp:turboq_prepare_query` | 100% — Q vectors still need `Q * q` |
| SDPA pipeline | `attn_quant_turboq.cpp:mha_turboq` | 100% — add codec dispatch in lambda bodies |
| KV chunk iteration | `turboq_foreach_kv_chunk` | 100% — same parallel chunking, prefetch, beam reorder |
| Softmax | `turboq_softmax` | 100% |
| Thread reduction | `turboq_reduce_head` | 100% — Q^T rotation applies identically (sign correction N/A) |
| Cache allocation | `scaled_attn.cpp` | Minimal changes (same byte sizes for dim=128) |

## New Files

| File | Purpose |
|------|---------|
| `tools/polar_codebook_gen.py` | Offline codebook generation for all levels and bit-widths |
| `polarq_tables.h` | Per-level codebook centroids, boundaries, cos/sin lookup tables |
| `attn_quant_polarq.hpp` | Public API: quantize, fused QK dot, V accum, batch wrappers |
| `attn_quant_polarq.cpp` | Implementation: scalar fallback + AVX-512 hot paths |

## Modified Files

| File | Path | Change |
|------|------|--------|
| `internal_properties.hpp` | `src/inference/dev_api/openvino/runtime/internal_properties.hpp` | Add `POLAR_QUANT_3`, `POLAR_QUANT_4` to `CacheCodecMode` enum (line ~182), add string parsing in `operator>>` (line ~208) |
| `config.cpp` | `src/plugins/intel_cpu/src/config.cpp` | Parse `"polar3"`, `"polar4"` strings |
| `plugin.cpp` | `src/plugins/intel_cpu/src/plugin.cpp` | Report polar codecs in `get_property` |
| `scaled_attn.cpp` | `src/plugins/intel_cpu/src/nodes/scaled_attn.cpp` | Add `is_polar_codec()` helper (~line 83), dispatch quantize + mha |
| `memory.cpp` | `src/plugins/intel_cpu/src/nodes/memory.cpp` | Extend `is_tbq` check (~line 1016) to include polar codecs for cache dimension sizing |
| `paged_attn.cpp` | `src/plugins/intel_cpu/src/nodes/paged_attn.cpp` | Extend TBQ rejection (~line 251) to also reject polar codecs (paged attention not supported) |
| `CMakeLists.txt` | `src/plugins/intel_cpu/CMakeLists.txt` | Add new source files, cross-compilation entries |
| `turboq_test.cpp` | `src/plugins/intel_cpu/tests/unit/turboq_test.cpp` | Add polar roundtrip, dot accuracy, SDPA comparison tests |

## Implementation Phases

### Phase 0: Codebook Generation (Python)

Create `tools/polar_codebook_gen.py`:
1. Generate 10M random dim-dimensional vectors (for dim=128 and dim=256)
2. Apply Haar rotation (same seed as C++ code)
3. Run recursive polar decomposition
4. Fit Lloyd-Max for each level at bit-widths 2, 3, 4, 5
   (codebooks are level-specific, not dim-specific — validate this assumption)
5. Compute cos/sin lookup tables for each codebook
6. Export as C header with `constexpr` arrays
7. Report per-level MSE for bit allocation optimization

### Phase 1: Tables Header

Create `polarq_tables.h`:
- Per-level angle codebooks (centroids)
- Per-level angle boundaries
- Pre-computed cos/sin lookup tables (indexed by quantized angle index)
- Bit allocation constants for polar3 and polar4
- `polarq_head_bytes(head_dim, bits)` function (not a constant, since it depends on dim)
- Static assertions for table consistency

### Phase 2: Quantize Kernel

Implement `polarq_quantize_head(src, dst, head_dim, bits)`:
- Norm extraction + unit vector normalization
- Rotation via existing `turboq_matvec`
- Iterative level-by-level polar decomposition (atan2 + sqrt)
- Per-level scalar quantization with level-specific boundaries
- Bit packing per level into the record layout

Scalar path only — not performance-critical (runs once per KV token during prefill).

### Phase 3: Fused QK Dot Kernel

Implement `polarq_fused_qk_dot(packed_k, q_rotated, head_dim, bits)`:
- Unpack angle indices per level
- Top-down radius reconstruction: cos/sin table lookups, multiply parent radii
- Level-1 partial dots: `r * (q[2i]*cos + q[2i+1]*sin)`
- Horizontal reduction

This is the hot path. Implement both scalar fallback and AVX-512:
- AVX-512: `_mm512_permutexvar_ps` for cos/sin table lookup (same pattern as TBQ codebook)
- Process each tree level vectorized across pairs within that level
- Level 1: 64 pairs = 4 × 16-wide SIMD iterations

### Phase 4: V Accumulation Kernel

Implement `polarq_fused_v_accum(packed_v, weights, accum_ptrs, ...)`:
- Same radius tree reconstruction as Phase 3
- Full Cartesian reconstruction at level 1
- Weighted accumulate into per-head accumulators

Also implement `polarq_batch_qk_dot` and `polarq_batch_v_accum` (batch wrappers
with prefetching, matching the TBQ batch pattern).

### Phase 5: SDPA Integration

- Add codec enum values and string parsing
- Add `is_polar_codec()`, `polar_bits()` helpers
- Dispatch to `polarq_quantize_head` in the KV cache write path
- Create `mha_polar` (or extend `mha_turboq`) for the attention path
- The 4-phase pipeline is identical; only the batch kernel lambdas change

### Phase 6: Functional Tests

Extend the existing functional test infrastructure to cover polar codecs.

**Correctness tests** — extend `ConcatSDPTurboQTest` in
`src/plugins/intel_cpu/tests/functional/custom/subgraph_tests/src/x64/concat_sdp_turboq.cpp`:

Add polar codec instantiations alongside existing TBQ ones. The test class
(`ConcatSDPTurboQTest`) is codec-agnostic — it builds a ReadValue/Gather/Concat/SDPA/Assign
subgraph and sets `KV_CACHE_CODEC` via the `m_kvCacheMode` string. Adding polar
codecs requires only new `INSTANTIATE_TEST_SUITE_P` entries:

```cpp
// dim=128
INSTANTIATE_TEST_SUITE_P(smoke_ConcatSDPPolarQTest,
                         ConcatSDPTurboQTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(turboqShapes_128),
                                            ::testing::Values("polar4", "polar3")),
                         ConcatSDPTurboQTest::getTestCaseName);

// dim=256
INSTANTIATE_TEST_SUITE_P(smoke_ConcatSDPPolarQTest_256,
                         ConcatSDPTurboQTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(turboqShapes_256),
                                            ::testing::Values("polar4", "polar3")),
                         ConcatSDPTurboQTest::getTestCaseName);
```

This gives 8 new tests (2 dims × 2 codecs × 2 batch sizes), run as:
```bash
bin/intel64/Release/ov_cpu_func_tests --gtest_filter="smoke_ConcatSDPPolarQ*"
```

The test compares quantized SDPA output against fp32 reference within the
existing tolerance threshold. If polar codecs need a different threshold,
parameterize it by codec mode in the test class.

**Benchmark tests** — extend `ConcatSDPKVBenchTest` in
`src/plugins/intel_cpu/tests/functional/custom/subgraph_tests/src/x64/concat_sdp_kv_bench.cpp`:

Add polar modes to the mode list and create benchmark instantiations:

```cpp
const std::vector<std::string> polarModes = {"polar4", "polar3"};

// Synthetic (H=Hk=8, head_dim=128)
INSTANTIATE_TEST_SUITE_P(smoke_KVCacheBench_polar,
                         ConcatSDPKVBenchTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(benchShapes_synthetic),
                                            ::testing::ValuesIn(polarModes),
                                            ::testing::Values(size_t{0})),
                         ConcatSDPKVBenchBase::getTestCaseName);

// Llama 3.1 8B (H=32, Hk=8, 4:1 GQA, head_dim=128)
INSTANTIATE_TEST_SUITE_P(smoke_KVCacheBench_llama8b_polar,
                         ConcatSDPKVBenchTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(benchShapes_llama8b),
                                            ::testing::ValuesIn(polarModes),
                                            ::testing::Values(size_t{0})),
                         ConcatSDPKVBenchBase::getTestCaseName);

// Gemma 3 27B (H=32, Hk=16, head_dim=256)
INSTANTIATE_TEST_SUITE_P(smoke_KVCacheBench_gemma3_polar,
                         ConcatSDPKVBenchTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(benchShapes_gemma3),
                                            ::testing::ValuesIn(polarModes),
                                            ::testing::Values(size_t{0})),
                         ConcatSDPKVBenchBase::getTestCaseName);
```

Benchmarks run at 20K context length (kWarmupIters=20000) to be memory-bound,
measuring steady-state decode us/step. Run as:
```bash
bin/intel64/Release/ov_cpu_func_tests --gtest_filter="*KVCacheBench*polar*"
```

Key metrics to compare against TBQ baselines:
- **us/step**: Polar QK dot latency vs TBQ (tree traversal overhead)
- **vs u8 ratio**: Relative to uniform u8 baseline
- **Memory**: Same record size → same RSS, so only compute time matters

**Modified test files:**

| File | Change |
|------|--------|
| `concat_sdp_turboq.cpp` | Add `smoke_ConcatSDPPolarQTest` and `_256` instantiations |
| `concat_sdp_kv_bench.cpp` | Add `polarModes` list and benchmark instantiations for synthetic, llama8b, gemma3 |

No changes needed to test class implementations (`concat_sdp_turboq.hpp`,
`concat_sdp_kv_bench.hpp`) — they are already codec-agnostic via the string parameter.

### Phase 7: AVX-512 Optimization + Performance

- Profile QK dot latency vs TBQ (target: within 1.5x of TBQ)
- Tree traversal has serial dependencies within levels but SIMD across pairs
- Optimize bit allocation using Phase 0's MSE data
- Consider FP16 norm (saves 2 bytes, frees bits for indices)

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| QK dot latency from tree traversal | Slower decode | Tree is only 7 levels deep; SIMD across pairs within each level |
| Complex bit packing (mixed widths) | Bug-prone | Reuse existing pack_3bit/pack_4bit where widths match; thorough roundtrip tests |
| Numerical error in radius product chain | Quality loss | Use precomputed cos/sin LUTs (full float precision); validate against exact dequant |
| Cache line crossing | Extra memory transaction | Polar4 (68 bytes) crosses a 64-byte cache line boundary, same as TBQ4. Prefetching (already in batch kernels) hides the cost. Polar3 (52 bytes) fits in one line. |
| Codebook quality | Suboptimal quantization | Validate with large sample size (10M+); compare against theoretical distributions |

## Expected Outcomes

Based on PolarQuant's theoretical advantages:
- **Polar4 should match or beat TBQ4** on all quality metrics (same byte budget,
  better bit allocation)
- **Polar3 should beat TBQ3** more significantly (lower levels are where bit
  allocation matters most)
- **QK dot latency** may be 1.2-1.5x higher than TBQ due to tree traversal
  (needs measurement)
- If Polar + QJL still underperforms Polar alone, it confirms QJL's variance
  problem is inherent to 1-bit sign correction, not a base-codec mismatch

## Comparison: Current TBQ vs PolarQuant

| Aspect | TBQ (current) | PolarQuant |
|--------|---------------|------------|
| Transform | Rotation only | Rotation + recursive polar decomposition |
| What's quantized | dim Cartesian coordinates | (dim-1) angles + 1 norm |
| Codebook | One per bit-width | One per (level, bit-width) pair |
| Bit allocation | Uniform across coordinates | Variable across levels |
| Level-1 angles | N/A | Uniform on [0, 2π) — easy to quantize |
| High-level angles | N/A | Concentrated at π/4 — few bits needed |
| Reconstruction | Codebook lookup | Tree of cos/sin lookups |
| Fused QK dot cost | dim lookups + dim FMAs | ~(dim-2) tree ops + dim FMAs |
| Norm storage | fp32 (4 bytes) | fp32 (4 bytes) |
| Record size (4-bit) | 68 bytes | 68 bytes |
| Record size (3-bit) | 52 bytes | 52 bytes |
