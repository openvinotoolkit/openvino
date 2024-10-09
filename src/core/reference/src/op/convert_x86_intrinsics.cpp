// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/convert.hpp"
#if defined(OV_CORE_USE_INTRINSICS)
#    include "openvino/reference/utils/convert_x86_intrinsics.hpp"

namespace ov {
namespace reference {

#    if defined(HAVE_AVX2)
template <>
__m128i NoClamp::apply<__m256i, __m128i>(const __m256i vec_i32) {
    // clang-format off
    static const auto shuffle = _mm256_setr_epi8(0,  4,  8, 12, -1, -1, -1, -1,  -1, -1, -1, -1, -1, -1, -1, -1,
                                                -1, -1, -1, -1,  0,  4,  8, 12,  -1, -1, -1, -1, -1, -1, -1, -1);
    // clang-format on

    const auto t = _mm256_shuffle_epi8(vec_i32, shuffle);
    const auto low = _mm256_castsi256_si128(t);
    const auto high = _mm256_extracti128_si256(t, 1);
    return _mm_or_si128(low, high);
}

template <>
template <>
__m256 Clamp<float, float16>::apply<__m256, __m256>(const __m256 vec_f32) {
    static const auto lo = _mm256_set1_ps(std::numeric_limits<float16>::lowest());
    static const auto hi = _mm256_set1_ps(std::numeric_limits<float16>::max());

    return _mm256_min_ps(_mm256_max_ps(vec_f32, lo), hi);
}

// --- f32 -> other
void Converter<float, float16>::Optimized<Clamp<float, float16>>::run(const float* in, float16* out) {
    auto vec_f32 = _mm256_loadu_ps(in);                                                        // load f32 input
    auto vec_f16 = _mm256_cvtps_ph(Clamp<float, float16>::apply<__m256, __m256>(vec_f32), 0);  // f32 -> f16 with clamp
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), vec_f16);                                // store f16 output
}

void Converter<float, float16>::Optimized<NoClamp>::run(const float* in, float16* out) {
    auto vec_f32 = _mm256_loadu_ps(in);                          // load f32 input
    auto vec_f16 = _mm256_cvtps_ph(vec_f32, _MM_ROUND_NEAREST);  // f32 -> f16
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), vec_f16);  // store f16 output
}

void Converter<float, int8_t>::Optimized<NoClamp>::run(const float* in, int8_t* out) {
    auto vec_f32 = _mm256_load_ps(in);                                 // load f32 input
    auto vec_i32 = _mm256_cvttps_epi32(vec_f32);                       // f32 -> i32
    auto vec_i8 = NoClamp::template apply<__m256i, __m128i>(vec_i32);  // i32 -> i8 no clamping
    _mm_storeu_si64(out, vec_i8);                                      // store i8 output
}

// --- f16 -> other
void Converter<float16, float>::Optimized<NoClamp>::run(const float16* in, float* out) {
    auto vec_f16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in));  // load input as f16 vector
    auto vec_f32 = _mm256_cvtph_ps(vec_f16);                               // convert f16 -> f32
    _mm256_storeu_ps(out, vec_f32);                                        // store f32 in output
}

void Converter<float16, int8_t>::Optimized<NoClamp>::run(const float16* in, int8_t* out) {
    const auto vec_f16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in));  // load input as f16 vector
    const auto vec_f32 = _mm256_cvtph_ps(vec_f16);                               // convert f16 -> f32
    auto vec_i32 = _mm256_cvttps_epi32(vec_f32);                                 // f32 -> i32
    auto vec_i8 = NoClamp::apply<__m256i, __m128i>(vec_i32);                     // i32 -> i8 no clamp
    _mm_storeu_si64(out, vec_i8);                                                // store i8 output
}

// --- bf16 -> other
void Converter<bfloat16, float16>::Optimized<Clamp<float, float16>>::run(const bfloat16* in, float16* out) {
    auto vec_bf16 = _mm256_cvtepu16_epi32(*reinterpret_cast<const __m128i*>(in));  // expand to 32-bits
    auto vec_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(vec_bf16, 16));           // shift left bf16 -> f32
    auto vec_f16 =
        _mm256_cvtps_ph(Clamp<float, float16>::apply<__m256, __m256>(vec_f32), _MM_ROUND_NEAREST);  // f32 -> f16
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), vec_f16);                                     // store f16
}

void Converter<bfloat16, float>::Optimized<NoClamp>::run(const bfloat16* in, float* out) {
    auto vec_f32 = _mm256_cvtepu16_epi32(*reinterpret_cast<const __m128i*>(in));  // expand to 32-bits
    vec_f32 = _mm256_slli_epi32(vec_f32, 16);                                     // shift left bf16 -> f32
    _mm256_storeu_ps(out, _mm256_castsi256_ps(vec_f32));                          // store f32 in output
}

// --- u8 -> other
void Converter<uint8_t, float16>::Optimized<NoClamp>::run(const uint8_t* in, float16* out) {
    auto i64 = _mm_loadu_si64(in);                               // load u8 input
    auto vec_i32 = _mm256_cvtepu8_epi32(i64);                    // u8 -> i32
    auto vec_f32 = _mm256_cvtepi32_ps(vec_i32);                  // i32 -> f32
    auto vec_f16 = _mm256_cvtps_ph(vec_f32, _MM_ROUND_NEAREST);  // f32 -> f16
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), vec_f16);  // store f16 output
}
#    endif  // HAVE_AVX2
}  // namespace reference
}  // namespace ov
#endif  // OV_CORE_USE_INTRINSICS
