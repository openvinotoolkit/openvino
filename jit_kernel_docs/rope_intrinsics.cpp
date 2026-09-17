// RoPE rotate_half, as a compiler baseline for the jit_kernel IR mode.
//
// Not built as part of the plugin and not called by anything: this is the
// reference `jit_kernel_validation.md` item 2 asks for — the same kernel
// taken through a full compiler, so the JIT's output can be compared
// against it instead of against an opinion. The numbers in
// `jit_kernel_journal.md` ("Reference: what a compiler produces" and
// "What is left against the compiler") come from here.
//
// Build and inspect:
//
//   g++     -O3 -march=native -mprefer-vector-width=512 -c rope_intrinsics.cpp -o ri_gcc.o
//   clang++ -O3 -march=native -mprefer-vector-width=512 -c rope_intrinsics.cpp -o ri_clang.o
//   objdump -d --no-show-raw-insn -M intel ri_gcc.o
//
// `-mprefer-vector-width=512` is not optional for a fair comparison:
// without it neither compiler uses zmm for an auto-vectorized loop on a
// Xeon, because default tuning caps at 256 bits. Any claim about beating
// or matching a compiler has to name the flags.
//
// The variants exist to separate what the compiler knows from what it can
// do: a compile-time count that divides the vector width, one that does
// not (where gcc picks a narrower remainder step by itself), a runtime
// count (where gcc gives up on vectorizing and clang emits 200
// instructions with 22 branches — the case a JIT wins by construction),
// and hand-written intrinsics for the masked and narrowed remainders.
//
// Same algorithm as jit_rotary_kernel::rotary_half:
//   dst[i]        = cos[i] * src[i] - sin[i] * src[i + half]
//   dst[i + half] = cos[i+half] * src[i+half] + sin[i+half] * src[i]

#include <immintrin.h>
#include <cstddef>

struct rotary_args {
    const float* src;
    const float* cos;
    const float* sin;
    float* dst;
};

// rotary_ndims=128, half=64, shift_cos_sin=true (cos/sin table is full-sized)
// 4 iterations of 16 floats = 64 elements per half
void rope_half_f32(const rotary_args* args) {
    const float* src = args->src;
    const float* cos = args->cos;
    const float* sin = args->sin;
    float* dst = args->dst;

    constexpr size_t half = 64;
    constexpr size_t N = 16;

    for (size_t i = 0; i < half; i += N) {
        __m512 v_src0 = _mm512_loadu_ps(src + i);
        __m512 v_src1 = _mm512_loadu_ps(src + i + half);
        __m512 v_cos  = _mm512_loadu_ps(cos + i);
        __m512 v_sin  = _mm512_loadu_ps(sin + i);

        // dst[i] = cos * src0 - sin * src1
        __m512 tmp = _mm512_mul_ps(v_sin, v_src1);
        __m512 dst0 = _mm512_fmsub_ps(v_cos, v_src0, tmp);
        _mm512_storeu_ps(dst + i, dst0);

        // Reload cos/sin for second half
        __m512 v_cos2 = _mm512_loadu_ps(cos + i + half);
        __m512 v_sin2 = _mm512_loadu_ps(sin + i + half);

        // dst[i + half] = cos2 * src1 + sin2 * src0
        __m512 tmp2 = _mm512_mul_ps(v_cos2, v_src1);
        __m512 dst1 = _mm512_fmadd_ps(v_sin2, v_src0, tmp2);
        _mm512_storeu_ps(dst + i + half, dst1);
    }
}

// Same but with variable count (tests tail handling)
void rope_half_f32_variable(const rotary_args* args, size_t half_rotary_ndims) {
    const float* src = args->src;
    const float* cos = args->cos;
    const float* sin = args->sin;
    float* dst = args->dst;

    constexpr size_t N = 16;

    for (size_t i = 0; i < half_rotary_ndims; i += N) {
        size_t remaining = half_rotary_ndims - i;
        __mmask16 mask = (remaining >= N) ? 0xFFFF : (1u << remaining) - 1;

        __m512 v_src0 = _mm512_maskz_loadu_ps(mask, src + i);
        __m512 v_src1 = _mm512_maskz_loadu_ps(mask, src + i + half_rotary_ndims);
        __m512 v_cos  = _mm512_maskz_loadu_ps(mask, cos + i);
        __m512 v_sin  = _mm512_maskz_loadu_ps(mask, sin + i);

        __m512 tmp = _mm512_mul_ps(v_sin, v_src1);
        __m512 dst0 = _mm512_fmsub_ps(v_cos, v_src0, tmp);
        _mm512_mask_storeu_ps(dst + i, mask, dst0);

        __m512 v_cos2 = _mm512_maskz_loadu_ps(mask, cos + i + half_rotary_ndims);
        __m512 v_sin2 = _mm512_maskz_loadu_ps(mask, sin + i + half_rotary_ndims);

        __m512 tmp2 = _mm512_mul_ps(v_cos2, v_src1);
        __m512 dst1 = _mm512_fmadd_ps(v_sin2, v_src0, tmp2);
        _mm512_mask_storeu_ps(dst + i + half_rotary_ndims, mask, dst1);
    }
}

// ── Added for the codegen comparison (2026-09-08) ─────────────────────
//
// Reference points for "what does the compiler produce", against the JIT:
//   legacy jit_rotary_kernel      446 B code, 92 insns, 0 branches (4x unrolled)
//   jit_kernel IR + mask folding  236 B code, 58 insns, 4 branches (rolled)
// QwenVL shape (half=40) is the interesting one: legacy handles the
// 8-element remainder with a *narrower* register (ymm), not a mask.

// 1. Plain C++, count known at compile time, multiple of 16.
//    Tests what the auto-vectorizer does with just the math.
void rope_half_plain_64(const rotary_args* args) {
    const float* __restrict src = args->src;
    const float* __restrict cos = args->cos;
    const float* __restrict sin = args->sin;
    float* __restrict dst = args->dst;
    constexpr size_t half = 64;

    for (size_t i = 0; i < half; ++i) {
        const float s0 = src[i];
        const float s1 = src[i + half];
        dst[i]        = cos[i] * s0 - sin[i] * s1;
        dst[i + half] = cos[i + half] * s1 + sin[i + half] * s0;
    }
}

// 2. Plain C++, count known at compile time, NOT a multiple of 16
//    (QwenVL). How does the compiler close out the remainder?
void rope_half_plain_40(const rotary_args* args) {
    const float* __restrict src = args->src;
    const float* __restrict cos = args->cos;
    const float* __restrict sin = args->sin;
    float* __restrict dst = args->dst;
    constexpr size_t half = 40;

    for (size_t i = 0; i < half; ++i) {
        const float s0 = src[i];
        const float s1 = src[i + half];
        dst[i]        = cos[i] * s0 - sin[i] * s1;
        dst[i + half] = cos[i + half] * s1 + sin[i + half] * s0;
    }
}

// 3. Plain C++, runtime count. The realistic shape for a kernel that is
//    not specialized per model.
void rope_half_plain_dynamic(const rotary_args* args, size_t half) {
    const float* __restrict src = args->src;
    const float* __restrict cos = args->cos;
    const float* __restrict sin = args->sin;
    float* __restrict dst = args->dst;

    for (size_t i = 0; i < half; ++i) {
        const float s0 = src[i];
        const float s1 = src[i + half];
        dst[i]        = cos[i] * s0 - sin[i] * s1;
        dst[i + half] = cos[i + half] * s1 + sin[i + half] * s0;
    }
}

namespace {

// One 16-lane step, predicated. mask == 0xffff for a full step.
inline void rope_step_masked(const float* src, const float* cos, const float* sin,
                             float* dst, size_t i, size_t half, __mmask16 mask) {
    const __m512 v_src0 = _mm512_maskz_loadu_ps(mask, src + i);
    const __m512 v_src1 = _mm512_maskz_loadu_ps(mask, src + i + half);
    const __m512 v_cos = _mm512_maskz_loadu_ps(mask, cos + i);
    const __m512 v_sin = _mm512_maskz_loadu_ps(mask, sin + i);

    _mm512_mask_storeu_ps(dst + i, mask,
                          _mm512_fmsub_ps(v_cos, v_src0, _mm512_mul_ps(v_sin, v_src1)));

    const __m512 v_cos2 = _mm512_maskz_loadu_ps(mask, cos + i + half);
    const __m512 v_sin2 = _mm512_maskz_loadu_ps(mask, sin + i + half);

    _mm512_mask_storeu_ps(dst + i + half, mask,
                          _mm512_fmadd_ps(v_sin2, v_src0, _mm512_mul_ps(v_cos2, v_src1)));
}

// One 8-lane step in ymm — the legacy kernel's trick for a remainder that
// happens to fill a narrower register exactly, so no predicate is needed.
inline void rope_step_ymm(const float* src, const float* cos, const float* sin,
                          float* dst, size_t i, size_t half) {
    const __m256 v_src0 = _mm256_loadu_ps(src + i);
    const __m256 v_src1 = _mm256_loadu_ps(src + i + half);
    const __m256 v_cos = _mm256_loadu_ps(cos + i);
    const __m256 v_sin = _mm256_loadu_ps(sin + i);

    _mm256_storeu_ps(dst + i,
                     _mm256_fmsub_ps(v_cos, v_src0, _mm256_mul_ps(v_sin, v_src1)));

    const __m256 v_cos2 = _mm256_loadu_ps(cos + i + half);
    const __m256 v_sin2 = _mm256_loadu_ps(sin + i + half);

    _mm256_storeu_ps(dst + i + half,
                     _mm256_fmadd_ps(v_sin2, v_src0, _mm256_mul_ps(v_cos2, v_src1)));
}

}  // namespace

// 4. half=40 with the remainder predicated: 2 full zmm steps + 1 masked.
//    This is what jit_kernel's mask tail folding emits, unrolled.
void rope_half_intrin_40_masked(const rotary_args* args) {
    constexpr size_t half = 40;
    for (size_t i = 0; i < half; i += 16) {
        const size_t remaining = half - i;
        const __mmask16 mask =
            remaining >= 16 ? __mmask16(0xffff) : __mmask16((1u << remaining) - 1u);
        rope_step_masked(args->src, args->cos, args->sin, args->dst, i, half, mask);
    }
}

// 5. half=40 with the remainder in ymm: 2 full zmm steps + 1 ymm step.
//    No predicate, no branch - the legacy kernel's shape.
void rope_half_intrin_40_narrow(const rotary_args* args) {
    constexpr size_t half = 40;
    rope_step_masked(args->src, args->cos, args->sin, args->dst, 0, half, 0xffff);
    rope_step_masked(args->src, args->cos, args->sin, args->dst, 16, half, 0xffff);
    rope_step_ymm(args->src, args->cos, args->sin, args->dst, 32, half);
}
