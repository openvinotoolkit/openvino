# SIMD Abstraction Layer

Namespace: `ov::Extensions::Cpu::XARCH::simd`

(`XARCH` is a macro set by OpenVINO's cross-compilation framework — it
resolves to `AVX2`, `AVX512F`, or `ANY` depending on the target ISA of each
translation unit.)

Compile-time abstraction over AVX-512 / AVX2 / scalar (and future ISAs).
`vec<T, i>` wraps the native register. Operators are members, everything
else is a free function found via ADL.

## Files

| File | Contents |
|------|----------|
| `simd_common.hpp` | `isa` enum, `active_isa`, primary templates for `vec`/`mask` |
| `simd_scalar.hpp` | Scalar specializations (always available, no `#ifdef`) |
| `simd_avx2.hpp` | AVX2 specializations (no `#ifdef` inside) |
| `simd_avx512.hpp` | AVX-512 specializations (no `#ifdef` inside) |
| `simd.hpp` | Aggregator: includes above + aliases (`f32`, `i32`), load API, `table` |

Each per-ISA header includes `simd_common.hpp` and is self-contained. The
only `#ifdef` is in `simd.hpp` — one conditional include per ISA.

### Known dependency

`simd_avx2.hpp` and `simd_avx512.hpp` depend on `scaled_attn/common.hpp`
for `mm256_loadu_u4_to_f32` / `mm512_loadu_u4_to_f32` (used in
`load_u4_pair`). Moving this logic into simd would make the directory
fully self-contained.

## Usage

```cpp
#include "nodes/kernels/simd/simd.hpp"

using namespace ov::Extensions::Cpu::XARCH;

simd::f32 a(1.0f);                          // broadcast
auto b = simd::load<simd::f32>(ptr);        // load (same-type or converting)
auto c = fmadd(a, b, simd::f32(0.0f));      // FMA (ADL from vec args)
store(c, out_ptr);                           // store (ADL)
float sum = reduce(c);                       // horizontal sum (ADL)

simd::table<16> cb(codebook_data);           // 16-entry LUT in registers
auto decoded = cb.lookup(indices);           // parallel LUT lookup
```

## Adding a New ISA

### Step 1: Add the enum value

In `simd_common.hpp`, add to the enum:

```cpp
enum class isa { scalar, avx2, avx512, neon /* new */ };
```

### Step 2: Create the per-ISA header

Create `simd_<name>.hpp` (e.g. `simd_neon.hpp`). Use any existing file as
template. The file must:

1. Include `simd_common.hpp`
2. Open `namespace ov::Extensions::Cpu::XARCH::simd`
3. Specialize `vec<float, isa::neon>` and `vec<int32_t, isa::neon>` with:
   - `using element_type = ...;`
   - `static constexpr int width = ...;`
   - `static constexpr isa isa_value = ...;`
   - Zero constructor, broadcast constructor, raw register constructor
   - `operator+`, `-`, `*` (float) and `operator&` (int32)
4. Provide all required free functions (see checklist below)
5. Close the namespace

No `#ifdef` guards inside the file.

### Step 3: Include from simd.hpp

Add one conditional include in `simd.hpp`:

```cpp
#if defined(HAVE_NEON)
#include "simd_neon.hpp"
#endif
```

### Step 4: Set active_isa

In `simd_common.hpp`, add the preprocessor branch:

```cpp
#if defined(HAVE_AVX512F)
inline constexpr isa active_isa = isa::avx512;
#elif defined(HAVE_AVX2)
inline constexpr isa active_isa = isa::avx2;
#elif defined(HAVE_NEON)          // new
inline constexpr isa active_isa = isa::neon;
#else
inline constexpr isa active_isa = isa::scalar;
#endif
```

### Free function checklist

Every per-ISA header must provide these free functions for `vec<float, isa::X>`:

**Required** (used by codec infrastructure):

| Function | Signature |
|----------|-----------|
| `store` | `void store(vec<float, I> v, float* p)` |
| `store` | `void store(vec<int32_t, I> v, int32_t* p)` |
| `reduce` | `float reduce(vec<float, I> v)` |
| `fmadd` | `vec<float, I> fmadd(vec<float, I> a, b, c)` |
| `load` (float) | `vec<float, I> load(const float* p, vec<float, I>*)` |
| `load` (f16) | `vec<float, I> load(const ov::float16* p, vec<float, I>*)` |
| `load` (bf16) | `vec<float, I> load(const ov::bfloat16* p, vec<float, I>*)` |
| `load` (u8→f32) | `vec<float, I> load(const uint8_t* p, vec<float, I>*)` |
| `load` (i32) | `vec<int32_t, I> load(const int32_t* p, vec<int32_t, I>*)` |
| `load` (u8→i32) | `vec<int32_t, I> load(const uint8_t* p, vec<int32_t, I>*)` |
| `partial_load` | `vec<float, I> partial_load(uint32_t k, const float* p, vec<float, I>*)` |
| `load_u4` | `vec<float, I> load_u4(const uint8_t* p, int bit_offset, vec<float, I>*)` |
| `load_u4_pair` | `void load_u4_pair(const uint8_t* p, vec<float, I>& lo, vec<float, I>& hi)` |
| `load_u8_pair` | `void load_u8_pair(const uint8_t* p, vec<float, I>& lo, vec<float, I>& hi)` |
| `permute` | `vec<float, I> permute(vec<float, I> table, vec<int32_t, I> idx)` |
| `srlv` | `vec<int32_t, I> srlv(vec<int32_t, I> val, vec<int32_t, I> shift)` |
| `select` | `vec<float, I> select(mask<I> m, vec<float, I> if_false, vec<float, I> if_true)` |
| comparisons | `mask<I> operator>(vec<int32_t, I>, vec<int32_t, I>)` (all 6) |

**Optional** (only needed for specific code paths):

| Function | Used by |
|----------|---------|
| `permute2` | `table::lookup` (N > W, AVX-512 path) |
| `unpack_lo/hi` | Polar interleave/deinterleave |
| `unpack_lo/hi_64` | AVX2 deinterleave |
| `shuffle<imm>` | AVX2 deinterleave |
| `permute_lanes<ctrl>` | AVX2 interleave |
| `permute_64<ctrl>` | AVX2 deinterleave |
| `broadcast_halves` | 3-bit unpack (AVX-512 only) |
| `select` (4-arg) | AVX2 compare+blend convenience |

Scalar stubs (assert-false or static_assert) are acceptable for optional
functions that are guarded by `if constexpr(i != isa::scalar)` at call sites.

### No changes needed in consumer code

The codec infrastructure (`codecs.hpp`, `turboq_codecs.hpp`, `polar_codecs.hpp`,
`attn_quant_turboq.cpp`) is fully generic over `vec<T, i>`. Adding a new ISA
requires zero changes outside this directory.
