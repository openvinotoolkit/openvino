// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <float.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include "attn_quant.hpp"
#include "attn_quant_kernel.hpp"
#include "common.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

using namespace ov;

template <typename T>
static void find_minmax(const T* src, size_t n, float& min, float& max) {
    max = -FLT_MAX;
    min = FLT_MAX;
    size_t i = 0;
#if defined(HAVE_AVX512F)
    auto v0_max = _mm512_set1_ps(-FLT_MAX);
    auto v0_min = _mm512_set1_ps(FLT_MAX);
    auto v1_max = _mm512_set1_ps(-FLT_MAX);
    auto v1_min = _mm512_set1_ps(FLT_MAX);
    auto v2_max = _mm512_set1_ps(-FLT_MAX);
    auto v2_min = _mm512_set1_ps(FLT_MAX);
    auto v3_max = _mm512_set1_ps(-FLT_MAX);
    auto v3_min = _mm512_set1_ps(FLT_MAX);
    for (; i + 4 * vec_len_f32_avx512 <= n; i += vec_len_f32_avx512 * 4) {
        auto v0 = mm512_uni_loadu_ps(src + i);
        auto v1 = mm512_uni_loadu_ps(src + i + vec_len_f32_avx512);
        auto v2 = mm512_uni_loadu_ps(src + i + 2 * vec_len_f32_avx512);
        auto v3 = mm512_uni_loadu_ps(src + i + 3 * vec_len_f32_avx512);
        v0_max = _mm512_max_ps(v0_max, v0);
        v0_min = _mm512_min_ps(v0_min, v0);
        v1_max = _mm512_max_ps(v1_max, v1);
        v1_min = _mm512_min_ps(v1_min, v1);
        v2_max = _mm512_max_ps(v2_max, v2);
        v2_min = _mm512_min_ps(v2_min, v2);
        v3_max = _mm512_max_ps(v3_max, v3);
        v3_min = _mm512_min_ps(v3_min, v3);
    }
    if (i + 2 * vec_len_f32_avx512 <= n) {
        auto v0 = mm512_uni_loadu_ps(src + i);
        auto v1 = mm512_uni_loadu_ps(src + i + vec_len_f32_avx512);
        v0_max = _mm512_max_ps(v0_max, v0);
        v0_min = _mm512_min_ps(v0_min, v0);
        v1_max = _mm512_max_ps(v1_max, v1);
        v1_min = _mm512_min_ps(v1_min, v1);
        i += 2 * vec_len_f32_avx512;
    }
    if (i + vec_len_f32_avx512 <= n) {
        auto v0 = mm512_uni_loadu_ps(src + i);
        v0_max = _mm512_max_ps(v0_max, v0);
        v0_min = _mm512_min_ps(v0_min, v0);
        i += vec_len_f32_avx512;
    }
    v0_max = _mm512_max_ps(v0_max, v1_max);
    v0_min = _mm512_min_ps(v0_min, v1_min);
    v2_max = _mm512_max_ps(v2_max, v3_max);
    v2_min = _mm512_min_ps(v2_min, v3_min);
    v0_max = _mm512_max_ps(v0_max, v2_max);
    v0_min = _mm512_min_ps(v0_min, v2_min);
    max = _mm512_reduce_max_ps(v0_max);
    min = _mm512_reduce_min_ps(v0_min);
#elif defined(HAVE_AVX2)
    auto v0_max = _mm256_set1_ps(-FLT_MAX);
    auto v0_min = _mm256_set1_ps(FLT_MAX);
    auto v1_max = _mm256_set1_ps(-FLT_MAX);
    auto v1_min = _mm256_set1_ps(FLT_MAX);
    auto v2_max = _mm256_set1_ps(-FLT_MAX);
    auto v2_min = _mm256_set1_ps(FLT_MAX);
    auto v3_max = _mm256_set1_ps(-FLT_MAX);
    auto v3_min = _mm256_set1_ps(FLT_MAX);
    for (; i + 4 * vec_len_f32_avx2 <= n; i += vec_len_f32_avx2 * 4) {
        auto v0 = mm256_uni_loadu_ps(src + i);
        auto v1 = mm256_uni_loadu_ps(src + i + vec_len_f32_avx2);
        auto v2 = mm256_uni_loadu_ps(src + i + 2 * vec_len_f32_avx2);
        auto v3 = mm256_uni_loadu_ps(src + i + 3 * vec_len_f32_avx2);
        v0_max = _mm256_max_ps(v0_max, v0);
        v0_min = _mm256_min_ps(v0_min, v0);
        v1_max = _mm256_max_ps(v1_max, v1);
        v1_min = _mm256_min_ps(v1_min, v1);
        v2_max = _mm256_max_ps(v2_max, v2);
        v2_min = _mm256_min_ps(v2_min, v2);
        v3_max = _mm256_max_ps(v3_max, v3);
        v3_min = _mm256_min_ps(v3_min, v3);
    }
    if (i + 2 * vec_len_f32_avx2 <= n) {
        auto v0 = mm256_uni_loadu_ps(src + i);
        auto v1 = mm256_uni_loadu_ps(src + i + vec_len_f32_avx2);
        v0_max = _mm256_max_ps(v0_max, v0);
        v0_min = _mm256_min_ps(v0_min, v0);
        v1_max = _mm256_max_ps(v1_max, v1);
        v1_min = _mm256_min_ps(v1_min, v1);
        i += 2 * vec_len_f32_avx2;
    }
    if (i + vec_len_f32_avx2 <= n) {
        auto v0 = mm256_uni_loadu_ps(src + i);
        v0_max = _mm256_max_ps(v0_max, v0);
        v0_min = _mm256_min_ps(v0_min, v0);
        i += vec_len_f32_avx2;
    }
    v0_max = _mm256_max_ps(v0_max, v1_max);
    v0_min = _mm256_min_ps(v0_min, v1_min);
    v2_max = _mm256_max_ps(v2_max, v3_max);
    v2_min = _mm256_min_ps(v2_min, v3_min);
    v0_max = _mm256_max_ps(v0_max, v2_max);
    v0_min = _mm256_min_ps(v0_min, v2_min);
    hmax(v0_max);
    hmin(v0_min);
    max = _mm256_cvtss_f32(v0_max);
    min = _mm256_cvtss_f32(v0_min);
#endif
    for (; i < n; i++) {
        float tmp = src[i];
        max = std::max(max, tmp);
        min = std::min(min, tmp);
    }
}

template <typename T>
static void quant_u8(const T* src, uint8_t* dst, size_t n, float& scale, float& zp) {
    size_t i = 0;
    float max = -FLT_MAX;
    float min = FLT_MAX;
    find_minmax(src, n, min, max);
    scale = (max - min) / 255;
    if (scale == 0)
        scale = 0.0001f;
    zp = -min / scale;
#if defined(HAVE_AVX512F)
    auto v_scale = _mm512_set1_ps(1 / scale);
    auto v_zp = _mm512_set1_ps(zp);
    auto v_zero = _mm512_setzero_epi32();
    for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
        auto v = mm512_uni_loadu_ps(src + i);
        v = _mm512_fmadd_ps(v, v_scale, v_zp);
        auto v_i32 = _mm512_cvt_roundps_epi32(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v_i32 = _mm512_max_epi32(v_i32, v_zero);
        _mm512_mask_cvtusepi32_storeu_epi8(dst + i, 0xffff, v_i32);
    }
#elif defined(HAVE_AVX2)
    auto v_scale = _mm256_set1_ps(1 / scale);
    auto v_zp = _mm256_set1_ps(zp);
    for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
        auto v = mm256_uni_loadu_ps(src + i);
        v = _mm256_fmadd_ps(v, v_scale, v_zp);
        v = _mm256_round_ps(v, _MM_ROUND_NEAREST);
        auto v_i32 = _mm256_cvtps_epi32(v);

        auto high4 = _mm256_extractf128_si256(v_i32, 1);
        auto low4 = _mm256_castsi256_si128(v_i32);
        auto packed = _mm_packs_epi32(low4, high4);
        packed = _mm_packus_epi16(packed, packed);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + i), packed);
    }
#endif
    for (; i < n; i++) {
        float tmp = src[i];
        dst[i] = static_cast<uint8_t>(std::round(tmp / scale + zp));
    }
}

template <typename T>
static void quant_u4(const T* src, void* dst, size_t n, float& scale, float& zp) {
    size_t i = 0;
    float max = -FLT_MAX;
    float min = FLT_MAX;
    find_minmax(src, n, min, max);
    auto insert_half_byte = [](uint8_t dst, uint8_t val, bool high_half) -> uint8_t {
        uint8_t shift = high_half ? 0 : 4;
        return dst | (uint8_t)(val << shift);
    };
    auto dst_ptr = reinterpret_cast<uint8_t*>(dst);
    scale = (max - min) / ((1 << 4) - 1);
    if (scale == 0)
        scale = 0.0001f;
    zp = -min / scale;
#if defined(HAVE_AVX512F)
    auto v_scale = _mm512_set1_ps(1 / scale);
    auto v_zp = _mm512_set1_ps(zp);
    auto v_zero = _mm512_setzero_epi32();
    auto v_upper = _mm512_set1_epi32(15);
    for (; i + 2 * vec_len_f32_avx512 <= n; i += 2 * vec_len_f32_avx512) {
        auto v0 = mm512_uni_loadu_ps(src + i);
        auto v1 = mm512_uni_loadu_ps(src + i + vec_len_f32_avx512);
        v0 = _mm512_fmadd_ps(v0, v_scale, v_zp);
        v1 = _mm512_fmadd_ps(v1, v_scale, v_zp);
        auto v0_i32 = _mm512_cvt_roundps_epi32(v0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        auto v1_i32 = _mm512_cvt_roundps_epi32(v1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v0_i32 = _mm512_max_epi32(v0_i32, v_zero);
        v1_i32 = _mm512_max_epi32(v1_i32, v_zero);
        v0_i32 = _mm512_min_epi32(v0_i32, v_upper);
        v1_i32 = _mm512_min_epi32(v1_i32, v_upper);
        __m512i idx1 = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
        __m512i idx2 = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
        auto first_half = _mm512_permutex2var_epi32(v0_i32, idx1, v1_i32);
        auto second_half = _mm512_permutex2var_epi32(v0_i32, idx2, v1_i32);
        first_half = _mm512_slli_epi32(first_half, 4);
        auto mask = _mm512_set1_epi32(0x0F);
        second_half = _mm512_and_epi32(second_half, mask);
        auto combined = _mm512_or_epi32(first_half, second_half);
        _mm512_mask_cvtepi32_storeu_epi8(dst_ptr + i / 2, 0xffff, combined);
    }
#endif
#if defined(HAVE_AVX2)
    auto v256_zero = _mm256_set1_epi32(0);
    auto v256_upper = _mm256_set1_epi32(15);
    auto v256_scale = _mm256_set1_ps(1 / scale);
    auto v256_zp = _mm256_set1_ps(zp);
    for (; i + vec_len_f32_avx2 * 2 <= n; i += vec_len_f32_avx2 * 2) {
        auto v0 = mm256_uni_loadu_ps(src + i);
        auto v1 = mm256_uni_loadu_ps(src + i + vec_len_f32_avx2);
        v0 = _mm256_fmadd_ps(v0, v256_scale, v256_zp);
        v1 = _mm256_fmadd_ps(v1, v256_scale, v256_zp);
        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);

        auto v0_i32 = _mm256_cvtps_epi32(v0);
        auto v1_i32 = _mm256_cvtps_epi32(v1);
        v0_i32 = _mm256_max_epi32(v0_i32, v256_zero);
        v1_i32 = _mm256_max_epi32(v1_i32, v256_zero);
        v0_i32 = _mm256_min_epi32(v0_i32, v256_upper);
        v1_i32 = _mm256_min_epi32(v1_i32, v256_upper);
        auto idx1 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
        v0_i32 = _mm256_permutevar8x32_epi32(v0_i32, idx1);
        v1_i32 = _mm256_permutevar8x32_epi32(v1_i32, idx1);
        //    0,1,2,3,4,5,6,7 | 8,9,10,11,12,13,14,15
        //       _mm256_permutevar8x32_epi32
        //    0,2,4,6,1,3,5,7 | 8,10,12,14,9,11,13,15
        //       _mm256_permute2x128_si256
        // 0,2,4,6,8,10,12,14 | 1,3,5,7,9,11,13,15
        //          shift + mask + or
        //     [0,1],[2,3], ..., [12,13], [14,15]
        auto first_half = _mm256_permute2x128_si256(v0_i32, v1_i32, 0x20);
        auto second_half = _mm256_permute2x128_si256(v0_i32, v1_i32, 0x31);
        first_half = _mm256_slli_epi32(first_half, 4);
        auto mask = _mm256_set1_epi32(0x0F);
        second_half = _mm256_and_si256(second_half, mask);
        auto combined = _mm256_or_si256(first_half, second_half);

        auto high4 = _mm256_extractf128_si256(combined, 1);
        auto low4 = _mm256_castsi256_si128(combined);
        // ignore sign bit for u4 case
        auto packed = _mm_packus_epi32(low4, high4);
        packed = _mm_packus_epi16(packed, packed);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst_ptr + i / 2), packed);
    }
#endif
    for (; i < n; i++) {
        float tmp = src[i];
#define MIN(a, b) ((a) < (b) ? (a) : (b))
        uint8_t src_val = MIN(15, (uint8_t)(std::round(tmp / scale + zp)));
        uint8_t dst_val = i % 2 == 0 ? 0 : dst_ptr[i / 2];
        dst_val = insert_half_byte(dst_val, src_val, (uint8_t)(i % 2));
        dst_ptr[i / 2] = dst_val;
    }
}

template <typename T,
          ov::element::Type_t DST_PREC,
          typename std::enable_if<DST_PREC == ov::element::u8, bool>::type = true>
static void quantize(const T* src, uint8_t* dst, size_t n, float* scale_zp) {
    quant_u8(src, dst, n, *scale_zp, *(scale_zp + 1));
}

template <typename T,
          ov::element::Type_t DST_PREC,
          typename std::enable_if<DST_PREC == ov::element::u4, bool>::type = true>
static void quantize(const T* src, void* dst, size_t n, float* scale_zp) {
    quant_u4(src, dst, n, *scale_zp, *(scale_zp + 1));
}

template <typename T, typename T2>
static void attn_quant_mt(const ov::intel_cpu::PlainTensor& k_src,
                          const ov::intel_cpu::PlainTensor& v_src,
                          const ov::intel_cpu::PlainTensor& k_dst,
                          const ov::intel_cpu::PlainTensor& v_dst,
                          const ov::intel_cpu::PlainTensor& k_scale_zp,
                          const ov::intel_cpu::PlainTensor& v_scale_zp) {
    // For compatibility, all input_kvs are permuted to BHLS
    size_t B = k_src.m_dims[0], H = k_src.m_dims[1], L1 = k_src.m_dims[2], S = k_src.m_dims[3], SV = v_src.m_dims[3];
    parallel_for3d(L1, B, H, [&](size_t m, size_t b, size_t h) {
        auto p_k = k_scale_zp.ptr<float>(m, b, h);
        auto p_v = v_scale_zp.ptr<float>(m, b, h);
        quant_u8(k_src.ptr<T>(b, h, m), k_dst.ptr<T2>(b, h, m), S, p_k[0], p_k[1]);
        quant_u8(v_src.ptr<T>(b, h, m), v_dst.ptr<T2>(b, h, m), SV, p_v[0], p_v[1]);
    });
}

template <typename T, ov::element::Type_t KEY_DST_PREC, ov::element::Type_t VALUE_DST_PREC>
static void paged_attn_quant_mt(const ov::intel_cpu::PlainTensor& k_src,
                                const ov::intel_cpu::PlainTensor& v_src,
                                const ov::intel_cpu::PlainTensor& k_dst,
                                const ov::intel_cpu::PlainTensor& v_dst,
                                const ov::intel_cpu::PlainTensor& slot_mapping,
                                const size_t key_group_size,
                                const size_t value_group_size) {
    size_t B = k_src.m_dims[0], H = k_src.m_dims[1], L1 = k_src.m_dims[2], S = k_src.m_dims[3], SV = v_src.m_dims[3];
    size_t block_size = k_dst.m_dims[2];
    size_t sub_byte_multiplier = 8 / v_dst.get_precision().bitwidth();
    parallel_for3d(B, L1, H, [&](size_t b, size_t m, size_t h) {
        auto slot = slot_mapping.ptr<int32_t>(b)[m];
        if (slot < 0)
            return;
        auto block_number = slot / block_size;
        auto block_offset = slot % block_size;
        // The layout for per token per head:
        // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized
        // feature(u8,idx_S)|
        for (size_t src_offset = 0, dst_offset = 0; src_offset < S;
             src_offset += key_group_size, dst_offset += key_group_size + sizeof(float) + sizeof(float)) {
            auto p_k = reinterpret_cast<float*>(
                k_dst.ptr<typename ov::element_type_traits<KEY_DST_PREC>::value_type>(block_number,
                                                                                      h,
                                                                                      block_offset,
                                                                                      dst_offset));
            quantize<T, KEY_DST_PREC>(
                k_src.ptr<T>(b, h, m, src_offset),
                k_dst.ptr<typename ov::element_type_traits<KEY_DST_PREC>::value_type>(block_number,
                                                                                      h,
                                                                                      block_offset,
                                                                                      dst_offset) +
                    sizeof(float) + sizeof(float),
                key_group_size,
                p_k);
        }

        for (size_t src_offset = 0, dst_offset = 0; src_offset < SV; src_offset += value_group_size,
                    dst_offset += value_group_size / sub_byte_multiplier + sizeof(float) + sizeof(float)) {
            uint8_t* v_base = reinterpret_cast<uint8_t*>(
                v_dst.m_ptr.get() +
                (block_number * v_dst.m_strides[0] + h * v_dst.m_strides[1] + block_offset * v_dst.m_strides[2]) /
                    sub_byte_multiplier +
                dst_offset);
            auto p_v = reinterpret_cast<float*>(v_base);
            uint8_t* v_ptr = v_base + sizeof(float) * 2;
            quantize<T, VALUE_DST_PREC>(v_src.ptr<T>(b, h, m, src_offset), v_ptr, value_group_size, p_v);
        }
    });
}

void attn_quantkv(const ov::intel_cpu::PlainTensor& k_src,
                  const ov::intel_cpu::PlainTensor& v_src,
                  const ov::intel_cpu::PlainTensor& k_dst,
                  const ov::intel_cpu::PlainTensor& v_dst,
                  const ov::intel_cpu::PlainTensor& k_scale_zp,
                  const ov::intel_cpu::PlainTensor& v_scale_zp) {
    if (k_src.get_precision() == ov::element::f32 && k_dst.get_precision() == ov::element::u8) {
        attn_quant_mt<float, uint8_t>(k_src, v_src, k_dst, v_dst, k_scale_zp, v_scale_zp);
    } else if (k_src.get_precision() == ov::element::bf16 && k_dst.get_precision() == ov::element::u8) {
        attn_quant_mt<ov::bfloat16, uint8_t>(k_src, v_src, k_dst, v_dst, k_scale_zp, v_scale_zp);
    } else if (k_src.get_precision() == ov::element::f16 && k_dst.get_precision() == ov::element::u8) {
        attn_quant_mt<ov::float16, uint8_t>(k_src, v_src, k_dst, v_dst, k_scale_zp, v_scale_zp);
    } else {
        OPENVINO_THROW("unsupport src type: ",
                       k_src.get_precision(),
                       ", dst type: ",
                       k_dst.get_precision(),
                       " in attn_quantkv");
    }
}

void paged_attn_quantkv(const ov::intel_cpu::PlainTensor& k_src,
                        const ov::intel_cpu::PlainTensor& v_src,
                        const ov::intel_cpu::PlainTensor& k_dst,
                        const ov::intel_cpu::PlainTensor& v_dst,
                        const ov::intel_cpu::PlainTensor& slot_mapping,
                        const size_t key_group_size,
                        const size_t value_group_size) {
    using function_type = void (*)(const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   const size_t,
                                   const size_t);
    static constexpr function_type funcs_fp32[] = {
        paged_attn_quant_mt<float, ov::element::u8, ov::element::u8>,
        paged_attn_quant_mt<float, ov::element::u8, ov::element::u4>,
    };
    static constexpr function_type funcs_bf16[] = {
        paged_attn_quant_mt<ov::bfloat16, ov::element::u8, ov::element::u8>,
        paged_attn_quant_mt<ov::bfloat16, ov::element::u8, ov::element::u4>,
    };
    static constexpr function_type funcs_f16[] = {
        paged_attn_quant_mt<ov::float16, ov::element::u8, ov::element::u8>,
        paged_attn_quant_mt<ov::float16, ov::element::u8, ov::element::u4>,
    };
    if (k_dst.get_precision() != ov::element::u8) {
        OPENVINO_THROW("unsupport src type: ",
                       k_src.get_precision(),
                       ", dst type: ",
                       k_dst.get_precision(),
                       " in paged_attn_quantkv");
    }
    std::map<ov::element::Type, size_t> dispatch_table = {
        {ov::element::u8, 0},
        {ov::element::u4, 1},
        {ov::element::i4, 2},
    };
    size_t dispatch = dispatch_table[v_dst.get_precision()];
    if (k_src.get_precision() == ov::element::f32) {
        funcs_fp32[dispatch](k_src, v_src, k_dst, v_dst, slot_mapping, key_group_size, value_group_size);
    } else if (k_src.get_precision() == ov::element::bf16) {
        funcs_bf16[dispatch](k_src, v_src, k_dst, v_dst, slot_mapping, key_group_size, value_group_size);
    } else if (k_src.get_precision() == ov::element::f16) {
        funcs_f16[dispatch](k_src, v_src, k_dst, v_dst, slot_mapping, key_group_size, value_group_size);
    }
}

void attn_quant_u8(const float* src, uint8_t* dst, size_t n, float& scale, float& zp) {
    quant_u8(src, dst, n, scale, zp);
}

void attn_dequant_u8(const uint8_t* src, float* dst, size_t n, float scale, float zp) {
    attn_dequant_kernel<float, ov::element::u8>(src, dst, n, scale, zp);
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov