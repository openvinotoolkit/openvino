// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// AVX2 ISA specializations for simd::vec, simd::mask, and free functions.
// Included only when HAVE_AVX2 is defined — no #ifdef guards inside.

#pragma once

#include <immintrin.h>

#include "nodes/kernels/scaled_attn/common.hpp"
#include "simd_common.hpp"

namespace ov::Extensions::Cpu::XARCH::simd {

// --- vec<float> ------------------------------------------------------------

template <>
struct vec<float, isa::avx2> {
    using element_type = float;
    static constexpr int width = 8;
    static constexpr isa isa_value = isa::avx2;
    __m256 v;

    vec() : v(_mm256_setzero_ps()) {}
    vec(float val) : v(_mm256_set1_ps(val)) {}  // NOLINT(google-explicit-constructor)
    vec(__m256 val) : v(val) {}                 // NOLINT(google-explicit-constructor)

    vec operator+(vec b) const {
        return {_mm256_add_ps(v, b.v)};
    }
    vec operator-(vec b) const {
        return {_mm256_sub_ps(v, b.v)};
    }
    vec operator*(vec b) const {
        return {_mm256_mul_ps(v, b.v)};
    }
};

inline void store(vec<float, isa::avx2> v, float* p) {
    _mm256_storeu_ps(p, v.v);
}
inline void store(vec<float, isa::avx2> v, ov::bfloat16* p) {
    // f32 → bf16: round-to-nearest-even, keep upper 16 bits.
    auto i32 = _mm256_castps_si256(v.v);
    auto lsb = _mm256_and_si256(_mm256_srli_epi32(i32, 16), _mm256_set1_epi32(1));
    auto rounded = _mm256_srli_epi32(_mm256_add_epi32(i32, _mm256_add_epi32(_mm256_set1_epi32(0x00007FFF), lsb)), 16);
    // Extract low 16 bits from each 32-bit lane without saturation via pshufb.
    const auto pack_mask = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    auto lo = _mm_shuffle_epi8(_mm256_castsi256_si128(rounded), pack_mask);
    auto hi = _mm_shuffle_epi8(_mm256_extracti128_si256(rounded, 1), pack_mask);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), _mm_unpacklo_epi64(lo, hi));
}
inline void store(vec<float, isa::avx2> v, ov::float16* p) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), _mm256_cvtps_ph(v.v, _MM_FROUND_TO_NEAREST_INT));
}
inline float reduce(vec<float, isa::avx2> v) {
    __m128 hi = _mm256_extractf128_ps(v.v, 1);
    __m128 lo = _mm256_castps256_ps128(v.v);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}
inline vec<float, isa::avx2> fmadd(vec<float, isa::avx2> a, vec<float, isa::avx2> b, vec<float, isa::avx2> c) {
    return {_mm256_fmadd_ps(a.v, b.v, c.v)};
}

// --- vec<int32_t> ----------------------------------------------------------

template <>
struct vec<int32_t, isa::avx2> {
    using element_type = int32_t;
    static constexpr int width = 8;
    static constexpr isa isa_value = isa::avx2;
    __m256i v;

    vec() : v(_mm256_setzero_si256()) {}
    vec(__m256i val) : v(val) {}                     // NOLINT(google-explicit-constructor)
    vec(int32_t val) : v(_mm256_set1_epi32(val)) {}  // NOLINT(google-explicit-constructor)

    static vec widen_u8(__m128i bytes) {
        return {_mm256_cvtepu8_epi32(bytes)};
    }
    vec operator&(vec b) const {
        return {_mm256_and_si256(v, b.v)};
    }
};

inline void store(vec<int32_t, isa::avx2> v, int32_t* p) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v.v);
}

// --- loads -----------------------------------------------------------------

inline vec<float, isa::avx2> load(const float* p, vec<float, isa::avx2>* /*tag*/) {
    return {_mm256_loadu_ps(p)};
}
inline vec<float, isa::avx2> load(const ov::float16* p, vec<float, isa::avx2>* /*tag*/) {
    return {_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)))};
}
inline vec<float, isa::avx2> load(const ov::bfloat16* p, vec<float, isa::avx2>* /*tag*/) {
    auto raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    return {_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16))};
}
inline vec<float, isa::avx2> load(const uint8_t* p, vec<float, isa::avx2>* /*tag*/) {
    return {_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))))};
}
inline vec<float, isa::avx2> partial_load(uint32_t k, const float* p, vec<float, isa::avx2>* /*tag*/) {
    const __m256i bit_masks = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    __m256i kmask =
        _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(static_cast<int32_t>(k)), bit_masks), bit_masks);
    return {_mm256_maskload_ps(p, kmask)};
}
inline vec<float, isa::avx2> load_u4(const uint8_t* p, int /*bit_offset*/, vec<float, isa::avx2>* /*tag*/) {
    const __m128i nibble_mask = _mm_set1_epi8(0x0F);
    __m128i raw = _mm_cvtsi32_si128(*reinterpret_cast<const int32_t*>(p));
    __m128i unpacked =
        _mm_unpacklo_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), nibble_mask), _mm_and_si128(raw, nibble_mask));
    return {_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(unpacked))};
}
inline void load_u4_pair(const uint8_t* p, vec<float, isa::avx2>& lo, vec<float, isa::avx2>& hi) {
    mm256_loadu_u4_to_f32(p, lo.v, hi.v);
}
inline void load_u8_pair(const uint8_t* p, vec<float, isa::avx2>& lo, vec<float, isa::avx2>& hi) {
    lo = load(p, static_cast<vec<float, isa::avx2>*>(nullptr));
    hi = load(p + 8, static_cast<vec<float, isa::avx2>*>(nullptr));
}

inline vec<int32_t, isa::avx2> load(const int32_t* p, vec<int32_t, isa::avx2>* /*tag*/) {
    return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))};
}
inline vec<int32_t, isa::avx2> load(const uint8_t* p, vec<int32_t, isa::avx2>* /*tag*/) {
    return {_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)))};
}

// --- arithmetic / permute --------------------------------------------------

inline vec<int32_t, isa::avx2> srlv(vec<int32_t, isa::avx2> val, vec<int32_t, isa::avx2> shift) {
    return {_mm256_srlv_epi32(val.v, shift.v)};
}

inline vec<float, isa::avx2> permute(vec<float, isa::avx2> table, vec<int32_t, isa::avx2> idx) {
    return {_mm256_permutevar8x32_ps(table.v, idx.v)};
}

inline vec<float, isa::avx2> unpack_lo(vec<float, isa::avx2> a, vec<float, isa::avx2> b) {
    return {_mm256_unpacklo_ps(a.v, b.v)};
}
inline vec<float, isa::avx2> unpack_hi(vec<float, isa::avx2> a, vec<float, isa::avx2> b) {
    return {_mm256_unpackhi_ps(a.v, b.v)};
}

template <int imm>
inline vec<float, isa::avx2> shuffle(vec<float, isa::avx2> a, vec<float, isa::avx2> b) {
    return {_mm256_shuffle_ps(a.v, b.v, imm)};
}

inline vec<float, isa::avx2> unpack_lo_64(vec<float, isa::avx2> a, vec<float, isa::avx2> b) {
    return {_mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(a.v), _mm256_castps_pd(b.v)))};
}
inline vec<float, isa::avx2> unpack_hi_64(vec<float, isa::avx2> a, vec<float, isa::avx2> b) {
    return {_mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(a.v), _mm256_castps_pd(b.v)))};
}

template <int ctrl>
inline vec<float, isa::avx2> permute_64(vec<float, isa::avx2> a) {
    return {_mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(a.v), ctrl))};
}

template <int ctrl>
inline vec<float, isa::avx2> permute_lanes(vec<float, isa::avx2> a, vec<float, isa::avx2> b) {
    return {_mm256_permute2f128_ps(a.v, b.v, ctrl)};
}

// --- mask ------------------------------------------------------------------

template <>
struct mask<isa::avx2> {
    __m256i v;
    mask() : v(_mm256_setzero_si256()) {}
    mask(__m256i val) : v(val) {}  // NOLINT(google-explicit-constructor)
};

inline mask<isa::avx2> operator>(vec<int32_t, isa::avx2> a, vec<int32_t, isa::avx2> b) {
    return {_mm256_cmpgt_epi32(a.v, b.v)};
}
inline mask<isa::avx2> operator<(vec<int32_t, isa::avx2> a, vec<int32_t, isa::avx2> b) {
    return b > a;
}
inline mask<isa::avx2> operator>=(vec<int32_t, isa::avx2> a, vec<int32_t, isa::avx2> b) {
    return {_mm256_or_si256(_mm256_cmpgt_epi32(a.v, b.v), _mm256_cmpeq_epi32(a.v, b.v))};
}
inline mask<isa::avx2> operator<=(vec<int32_t, isa::avx2> a, vec<int32_t, isa::avx2> b) {
    return b >= a;
}
inline mask<isa::avx2> operator==(vec<int32_t, isa::avx2> a, vec<int32_t, isa::avx2> b) {
    return {_mm256_cmpeq_epi32(a.v, b.v)};
}
inline mask<isa::avx2> operator!=(vec<int32_t, isa::avx2> a, vec<int32_t, isa::avx2> b) {
    return {_mm256_xor_si256(_mm256_cmpeq_epi32(a.v, b.v), _mm256_set1_epi32(-1))};
}

inline vec<float, isa::avx2> select(mask<isa::avx2> m, vec<float, isa::avx2> if_false, vec<float, isa::avx2> if_true) {
    return {_mm256_blendv_ps(if_false.v, if_true.v, _mm256_castsi256_ps(m.v))};
}

inline vec<float, isa::avx2> select(vec<int32_t, isa::avx2> val,
                                    vec<int32_t, isa::avx2> threshold,
                                    vec<float, isa::avx2> if_false,
                                    vec<float, isa::avx2> if_true) {
    return select(val > threshold, if_false, if_true);
}

}  // namespace ov::Extensions::Cpu::XARCH::simd
