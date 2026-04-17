// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// AVX-512 ISA specializations for simd::vec, simd::mask, and free functions.
// Included only when HAVE_AVX512F is defined — no #ifdef guards inside.

#pragma once

#include <immintrin.h>

// @todo get rid of this dependency by moving logic from common.hpp to simd and use simd everywhere
// this will also make simd selfcontained and allow to write perfect unit tests
#include "nodes/kernels/scaled_attn/common.hpp"
#include "simd_common.hpp"

namespace ov::Extensions::Cpu::XARCH::simd {

// --- vec<float> ------------------------------------------------------------

template <>
struct vec<float, isa::avx512> {
    using element_type = float;
    static constexpr int width = 16;
    static constexpr isa isa_value = isa::avx512;
    __m512 v;

    vec() : v(_mm512_setzero_ps()) {}
    vec(float val) : v(_mm512_set1_ps(val)) {}  // NOLINT(google-explicit-constructor)
    vec(__m512 val) : v(val) {}                 // NOLINT(google-explicit-constructor)

    vec operator+(vec b) const {
        return {_mm512_add_ps(v, b.v)};
    }
    vec operator-(vec b) const {
        return {_mm512_sub_ps(v, b.v)};
    }
    vec operator*(vec b) const {
        return {_mm512_mul_ps(v, b.v)};
    }
};

inline void store(vec<float, isa::avx512> v, float* p) {
    _mm512_storeu_ps(p, v.v);
}
inline void store(vec<float, isa::avx512> v, ov::bfloat16* p) {
    // f32 → bf16: truncate each f32 to upper 16 bits (round-to-nearest-even via shift).
    auto i32 = _mm512_castps_si512(v.v);
    auto u16 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_add_epi32(i32, _mm512_set1_epi32(0x00008000)), 16));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), u16);
}
inline void store(vec<float, isa::avx512> v, ov::float16* p) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), _mm512_cvtps_ph(v.v, _MM_FROUND_TO_NEAREST_INT));
}
inline float reduce(vec<float, isa::avx512> v) {
    return _mm512_reduce_add_ps(v.v);
}
inline vec<float, isa::avx512> fmadd(vec<float, isa::avx512> a, vec<float, isa::avx512> b, vec<float, isa::avx512> c) {
    return {_mm512_fmadd_ps(a.v, b.v, c.v)};
}

// --- vec<int32_t> ----------------------------------------------------------

template <>
struct vec<int32_t, isa::avx512> {
    using element_type = int32_t;
    static constexpr int width = 16;
    static constexpr isa isa_value = isa::avx512;
    __m512i v;

    vec() : v(_mm512_setzero_si512()) {}
    vec(__m512i val) : v(val) {}                     // NOLINT(google-explicit-constructor)
    vec(int32_t val) : v(_mm512_set1_epi32(val)) {}  // NOLINT(google-explicit-constructor)

    static vec widen_u8(__m128i bytes) {
        return {_mm512_cvtepu8_epi32(bytes)};
    }
    vec operator&(vec b) const {
        return {_mm512_and_epi32(v, b.v)};
    }

    static vec broadcast_halves(int32_t lo, int32_t hi) {
        return {_mm512_mask_set1_epi32(_mm512_set1_epi32(lo), 0xFF00, hi)};
    }
};

inline void store(vec<int32_t, isa::avx512> v, int32_t* p) {
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(p), v.v);
}

// --- loads -----------------------------------------------------------------

inline vec<float, isa::avx512> load(const float* p, vec<float, isa::avx512>* /*tag*/) {
    return {_mm512_loadu_ps(p)};
}
inline vec<float, isa::avx512> load(const ov::float16* p, vec<float, isa::avx512>* /*tag*/) {
    return {_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)))};
}
inline vec<float, isa::avx512> load(const ov::bfloat16* p, vec<float, isa::avx512>* /*tag*/) {
    auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    return {_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16))};
}
inline vec<float, isa::avx512> load(const uint8_t* p, vec<float, isa::avx512>* /*tag*/) {
    return {_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))))};
}
inline vec<float, isa::avx512> partial_load(uint32_t k, const float* p, vec<float, isa::avx512>* /*tag*/) {
    return {_mm512_maskz_loadu_ps(static_cast<__mmask16>(k), p)};
}
inline vec<float, isa::avx512> load_u4(const uint8_t* p, int /*bit_offset*/, vec<float, isa::avx512>* /*tag*/) {
    const __m128i nibble_mask = _mm_set1_epi8(0x0F);
    __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    __m128i unpacked =
        _mm_unpacklo_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), nibble_mask), _mm_and_si128(raw, nibble_mask));
    return {_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(unpacked))};
}
inline void load_u4_pair(const uint8_t* p, vec<float, isa::avx512>& lo, vec<float, isa::avx512>& hi) {
    mm512_loadu_u4_to_f32(p, lo.v, hi.v);
}
inline void load_u8_pair(const uint8_t* p, vec<float, isa::avx512>& lo, vec<float, isa::avx512>& hi) {
    lo = load(p, static_cast<vec<float, isa::avx512>*>(nullptr));
    hi = load(p + 16, static_cast<vec<float, isa::avx512>*>(nullptr));
}

inline vec<int32_t, isa::avx512> load(const int32_t* p, vec<int32_t, isa::avx512>* /*tag*/) {
    return {_mm512_loadu_si512(reinterpret_cast<const __m512i*>(p))};
}
inline vec<int32_t, isa::avx512> load(const uint8_t* p, vec<int32_t, isa::avx512>* /*tag*/) {
    return {_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)))};
}

// --- arithmetic / permute --------------------------------------------------

inline vec<int32_t, isa::avx512> srlv(vec<int32_t, isa::avx512> val, vec<int32_t, isa::avx512> shift) {
    return {_mm512_srlv_epi32(val.v, shift.v)};
}

inline vec<float, isa::avx512> permute(vec<float, isa::avx512> table, vec<int32_t, isa::avx512> idx) {
    return {_mm512_permutexvar_ps(idx.v, table.v)};
}

inline vec<float, isa::avx512> permute2(vec<float, isa::avx512> table_lo,
                                        vec<int32_t, isa::avx512> idx,
                                        vec<float, isa::avx512> table_hi) {
    return {_mm512_permutex2var_ps(table_lo.v, idx.v, table_hi.v)};
}

template <int imm>
inline vec<float, isa::avx512> shuffle(vec<float, isa::avx512> a, vec<float, isa::avx512> b) {
    return {_mm512_shuffle_ps(a.v, b.v, imm)};
}

inline vec<float, isa::avx512> unpack_lo(vec<float, isa::avx512> a, vec<float, isa::avx512> b) {
    return {_mm512_unpacklo_ps(a.v, b.v)};
}
inline vec<float, isa::avx512> unpack_hi(vec<float, isa::avx512> a, vec<float, isa::avx512> b) {
    return {_mm512_unpackhi_ps(a.v, b.v)};
}

// --- mask ------------------------------------------------------------------

template <>
struct mask<isa::avx512> {
    __mmask16 v;
    mask() : v(0) {}
    mask(__mmask16 val) : v(val) {}  // NOLINT(google-explicit-constructor)
};

inline mask<isa::avx512> operator>(vec<int32_t, isa::avx512> a, vec<int32_t, isa::avx512> b) {
    return {_mm512_cmpgt_epi32_mask(a.v, b.v)};
}
inline mask<isa::avx512> operator<(vec<int32_t, isa::avx512> a, vec<int32_t, isa::avx512> b) {
    return {_mm512_cmplt_epi32_mask(a.v, b.v)};
}
inline mask<isa::avx512> operator>=(vec<int32_t, isa::avx512> a, vec<int32_t, isa::avx512> b) {
    return {_mm512_cmpge_epi32_mask(a.v, b.v)};
}
inline mask<isa::avx512> operator<=(vec<int32_t, isa::avx512> a, vec<int32_t, isa::avx512> b) {
    return {_mm512_cmple_epi32_mask(a.v, b.v)};
}
inline mask<isa::avx512> operator==(vec<int32_t, isa::avx512> a, vec<int32_t, isa::avx512> b) {
    return {_mm512_cmpeq_epi32_mask(a.v, b.v)};
}
inline mask<isa::avx512> operator!=(vec<int32_t, isa::avx512> a, vec<int32_t, isa::avx512> b) {
    return {_mm512_cmpneq_epi32_mask(a.v, b.v)};
}

inline vec<float, isa::avx512> select(mask<isa::avx512> m,
                                      vec<float, isa::avx512> if_false,
                                      vec<float, isa::avx512> if_true) {
    return {_mm512_mask_blend_ps(m.v, if_false.v, if_true.v)};
}

}  // namespace ov::Extensions::Cpu::XARCH::simd
