// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cassert>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

// avx512/avx2 register length in byte
static constexpr size_t vec_len_avx512 = 64lu;
static constexpr size_t vec_len_avx2 = 32lu;
static constexpr size_t vec_len_neon = 16lu;
// avx512/avx2 register length in float
static constexpr size_t vec_len_f32_avx512 = vec_len_avx512 / sizeof(float);
static constexpr size_t vec_len_f32_avx2 = vec_len_avx2 / sizeof(float);
static constexpr size_t vec_len_f32_neon = vec_len_neon / sizeof(float);

#ifdef HAVE_AVX512F
    inline __m512 cvt_bf16_to_fp32(const __m256i src) {
        __m512i y = _mm512_cvtepu16_epi32(src);
        return _mm512_castsi512_ps(_mm512_slli_epi32(y, 16));
    }

    inline __m512 mm512_uni_loadu_ps(const ov::bfloat16* a) {
        auto vec_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
        return cvt_bf16_to_fp32(vec_bf16);
    }

    inline __m512 mm512_uni_loadu_ps(const float* a) {
        return _mm512_loadu_ps(a);
    }

    inline __m512 mm512_uni_loadu_tail_ps(const float* a, size_t count) {
        __mmask16 mask = (1 << count) - 1;
        return _mm512_maskz_loadu_ps(mask, a);
    }

    inline __m512 mm512_uni_loadu_tail_ps(const ov::bfloat16* a, size_t count) {
        auto mask = (1 << count) - 1;
        auto bf16_vec = _mm256_maskz_loadu_epi16(mask, a);
        return cvt_bf16_to_fp32(bf16_vec);
    }

    inline void mm512_uni_storeu_ps(float* a,  __m512 v) {
        _mm512_storeu_ps(a, v);
    }
    inline void mm512_uni_storeu_ps(ov::bfloat16 *addr, __m512 xps) {
        __m512i xpi32 = _mm512_castps_si512(xps);
        __m512i nan = _mm512_set1_epi32(0xffff);
        auto mask = _mm512_cmp_ps_mask(xps, xps, _CMP_ORD_Q);
        __m512i ones = _mm512_set1_epi32(0x1);
        __m512i vec_bias = _mm512_set1_epi32(0x7fff);
        auto x = _mm512_and_si512(_mm512_srli_epi32(xpi32, 16), ones); // LSB = x[16]
        x = _mm512_add_epi32(x, vec_bias);                             // rounding_bias = 0x7fff + LSB
        x = _mm512_srli_epi32(_mm512_add_epi32(x, xpi32), 16);         // x = (x + rounding_bias) >> 16;
        x = _mm512_mask_blend_epi32(mask, nan, x);                     // Check NaN before converting back to bf16
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(addr), _mm512_cvtepi32_epi16(x));
    }
    inline void mm512_uni_mask_storeu_ps(ov::bfloat16 *addr, __mmask16 mask_addr, __m512 xps) {
        __m512i xpi32 = _mm512_castps_si512(xps);
        __m512i nan = _mm512_set1_epi32(0xffff);
        auto mask = _mm512_cmp_ps_mask(xps, xps, _CMP_ORD_Q);
        __m512i ones = _mm512_set1_epi32(0x1);
        __m512i vec_bias = _mm512_set1_epi32(0x7fff);
        auto x = _mm512_and_si512(_mm512_srli_epi32(xpi32, 16), ones); // LSB = x[16]
        x = _mm512_add_epi32(x, vec_bias);                             // rounding_bias = 0x7fff + LSB
        x = _mm512_srli_epi32(_mm512_add_epi32(x, xpi32), 16);         // x = (x + rounding_bias) >> 16;
        x = _mm512_mask_blend_epi32(mask, nan, x);                     // Check NaN before converting back to bf16
        _mm512_mask_cvtepi32_storeu_epi16(addr, mask_addr, x);
    }

    inline __m512 mm512_uni_loadu_ps(ov::float16* a) {
        auto vec_f16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
        return _mm512_cvtph_ps(vec_f16);
    }
    inline __m512 mm512_uni_loadu_tail_ps(const ov::float16* a, size_t count) {
        auto mask = (1 << count) - 1;
        auto f16_vec = _mm256_maskz_loadu_epi16(mask, a);
        return _mm512_cvtph_ps(f16_vec);
    }
    inline void mm512_uni_storeu_ps(ov::float16* addr,  __m512 v) {
        __m256i vec_f16 = _mm512_cvtps_ph(v, 0);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(addr), vec_f16);
    }
#endif

#ifdef HAVE_AVX2
    inline __m256i get_mask(int N7) {
        static __m256i mask[] = {
            _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0),
            _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1),
            _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1),
            _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1),
            _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1),
            _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1),
            _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1),
            _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1),
            _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1),
        };
        return _mm256_loadu_si256(&mask[N7]);
    }
    inline __m256 mm256_uni_loadu_ps(const float* a) {
        return _mm256_loadu_ps(a);
    }
    inline void mm256_uni_storeu_ps(float* a,  __m256 v) {
        _mm256_storeu_ps(a, v);
    }

    inline __m256 mm256_uni_loadu_ps(const ov::bfloat16* a) {
        auto vec_bf16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
        auto o = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(vec_bf16), 16));
        return o;
    }

    inline __m256 mm256_uni_loadu_tail_ps(const float* a, const size_t count) {
        auto mask = get_mask(count);
        return _mm256_maskload_ps(a, mask);
    }

    inline __m256 mm256_uni_loadu_tail_ps(const ov::bfloat16* a, const size_t count) {
        assert("AVX2 version of bfloat16 tail load is just for compilation pass");
        ov::bfloat16 tmp_values[8] = {0};
        std::memcpy(tmp_values, a, count * sizeof(ov::bfloat16));
        return mm256_uni_loadu_ps(tmp_values);
    }

    inline void mm256_uni_storeu_ps(ov::bfloat16 *addr, __m256 xps) {
        __m256i xpi32 = _mm256_castps_si256(xps);
        __m256i nan = _mm256_set1_epi32(0xffff);
        __m256i mask = _mm256_castps_si256(_mm256_cmp_ps(xps, xps, _CMP_ORD_Q));
        __m256i ones = _mm256_set1_epi32(0x1);
        __m256i vec_bias = _mm256_set1_epi32(0x7fff);
        auto x = _mm256_and_si256(_mm256_srli_epi32(xpi32, 16), ones); // LSB = x[16]
        x = _mm256_add_epi32(x, vec_bias);                             // rounding_bias = 0x7fff + LSB
        x = _mm256_srli_epi32(_mm256_add_epi32(x, xpi32), 16);         // x = (x + rounding_bias) >> 16;
        x = _mm256_blendv_epi8(nan, x, mask);                          // Check NaN before converting back to bf16
        x = _mm256_packus_epi32(x, x);
        x = _mm256_permute4x64_epi64(x, 0xd8);
        __m128i bf16_o = _mm256_extractf128_si256(x, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(addr), bf16_o);
    }

    inline __m256 mm256_uni_loadu_ps(ov::float16* a) {
        auto vec_f16 = _mm_loadu_si128(reinterpret_cast<__m128i*>(a));
        auto o = _mm256_cvtph_ps(vec_f16);
        return o;
    }
    inline __m256 mm256_uni_loadu_tail_ps(const ov::float16* a, const size_t count) {
        ov::float16 tmp_values[8] = {0};
        std::memcpy(tmp_values, a, count * sizeof(ov::float16));
        return mm256_uni_loadu_ps(tmp_values);
    }
    inline void mm256_uni_storeu_ps(ov::float16* a,  __m256 v) {
        __m128i vec_f16 = _mm256_cvtps_ph(v, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(a), vec_f16);
    }

    inline void hsum(__m256& x) {
        __m256 y;                             // x:  0 1 2 3   4 5 6 7
        y = _mm256_permute_ps(x, 0x39);       // y:  1 2 3 0   5 6 7 4
        x = _mm256_add_ps(x, y);              // X:  01 12 23 30  45 56 67 74
        y = _mm256_permute_ps(x, 0x4e);       // y:  23 30 01 12  67 74 45 56
        x = _mm256_add_ps(x, y);              // x: 0123 x x x   4567 x x x
        y = _mm256_permute2f128_ps(x, x, 1);  // y: 4567 x x x  0123 x x x
        x = _mm256_add_ps(x, y);              // x: 01234567 x x x x x x x
    }
    inline void hmax(__m256& x) {
        __m256 y;                             // x:  0 1 2 3   4 5 6 7
        y = _mm256_permute_ps(x, 0x39);       // y:  1 2 3 0   5 6 7 4
        x = _mm256_max_ps(x, y);              // X:  01 12 23 30  45 56 67 74
        y = _mm256_permute_ps(x, 0x4e);       // y:  23 30 01 12  67 74 45 56
        x = _mm256_max_ps(x, y);              // x: 0123 x x x   4567 x x x
        y = _mm256_permute2f128_ps(x, x, 1);  // y: 4567 x x x  0123 x x x
        x = _mm256_max_ps(x, y);              // x: 01234567 x x x x x x x
    }
    inline void hmin(__m256& x) {
        __m256 y;                             // x:  0 1 2 3   4 5 6 7
        y = _mm256_permute_ps(x, 0x39);       // y:  1 2 3 0   5 6 7 4
        x = _mm256_min_ps(x, y);              // X:  01 12 23 30  45 56 67 74
        y = _mm256_permute_ps(x, 0x4e);       // y:  23 30 01 12  67 74 45 56
        x = _mm256_min_ps(x, y);              // x: 0123 x x x   4567 x x x
        y = _mm256_permute2f128_ps(x, x, 1);  // y: 4567 x x x  0123 x x x
        x = _mm256_min_ps(x, y);              // x: 01234567 x x x x x x x
    }
#endif

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov