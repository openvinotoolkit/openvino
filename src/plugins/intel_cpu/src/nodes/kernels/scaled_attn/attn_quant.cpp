// Copyright (C) 2018-2024 Intel Corporation
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
static void quant_u8(const T* src, uint8_t* dst, size_t n, float& scale, float& zp) {
    size_t i = 0;
    float max = -FLT_MAX;
    float min = FLT_MAX;
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
    scale = (max - min) / 255;
    if (scale == 0)
        scale = 0.0001f;
    zp = -min / scale;

    i = 0;
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
static void quant_u8_by_channel_kernel(const T* src,
                                       uint8_t* dst,
                                       size_t seq_dim,
                                       size_t hidden_dims,
                                       size_t src_stride,
                                       size_t dst_stride,
                                       float* scale,
                                       float* zp) {
    size_t j = 0;
#if defined(HAVE_AVX512F)
    for (; j + vec_len_f32_avx512 <= hidden_dims; j += vec_len_f32_avx512) {
        auto v_max = _mm512_set1_ps(-std::numeric_limits<float>::max());
        auto v_min = _mm512_set1_ps(std::numeric_limits<float>::max());
        for (size_t i = 0; i < seq_dim; i += 1) {
            auto v_cur = mm512_uni_loadu_ps(src + i * src_stride + j);
            v_max = _mm512_max_ps(v_max, v_cur);
            v_min = _mm512_min_ps(v_min, v_cur);
        }
        auto v_scale = _mm512_sub_ps(v_max, v_min);
        v_scale = _mm512_mul_ps(v_scale, _mm512_set1_ps(1.0f / 255));
        auto v_mask = _mm512_cmp_ps_mask(v_scale, _mm512_setzero_ps(), _CMP_EQ_OQ);
        v_scale = _mm512_mask_add_ps(v_scale, v_mask, v_scale, _mm512_set1_ps(0.0001f));
        auto v_zp = _mm512_mul_ps(v_min, _mm512_set1_ps(-1.0f));
        v_zp = _mm512_div_ps(v_zp, v_scale);

        _mm512_storeu_ps(scale + j, v_scale);
        _mm512_storeu_ps(zp + j, v_zp);
    }
#endif
#if defined(HAVE_AVX2)
    for (; j + vec_len_f32_avx2 <= hidden_dims; j += vec_len_f32_avx2) {
        auto v_max = _mm256_set1_ps(-std::numeric_limits<float>::max());
        auto v_min = _mm256_set1_ps(std::numeric_limits<float>::max());
        for (size_t i = 0; i < seq_dim; i++) {
            auto v_cur = mm256_uni_loadu_ps(src + i * src_stride + j);
            v_max = _mm256_max_ps(v_max, v_cur);
            v_min = _mm256_min_ps(v_min, v_cur);
        }
        auto v_scale = _mm256_sub_ps(v_max, v_min);
        v_scale = _mm256_mul_ps(v_scale, _mm256_set1_ps(1 / 255.0f));
        auto v_cond = _mm256_cmp_ps(v_scale, _mm256_setzero_ps(), _CMP_EQ_OQ);
        auto v_comp = _mm256_and_ps(v_cond, _mm256_set1_ps(0.0001f));
        v_scale = _mm256_add_ps(v_scale, v_comp);
        auto v_zp = _mm256_mul_ps(v_min, _mm256_set1_ps(-1.0f));
        v_zp = _mm256_div_ps(v_zp, v_scale);
        _mm256_storeu_ps(scale + j, v_scale);
        _mm256_storeu_ps(zp + j, v_zp);
    }
#endif
    for (; j < hidden_dims; j++) {
        float max = -std::numeric_limits<float>::max();
        float min = std::numeric_limits<float>::max();
        for (size_t i = 0; i < seq_dim; i++) {
            float tmp = src[i * src_stride + j];
            max = std::max(max, tmp);
            min = std::min(min, tmp);
        }
        float orgin_range = (max - min);
        float temp_scale = 0.0f;
        float temp_zp = 0.0f;
        if (orgin_range != 0.0f) {
            temp_scale = orgin_range / 255;
            temp_zp = -255 * min / orgin_range;
        } else {
            temp_scale = 0.0001f;
            temp_zp = -min / temp_scale;
        }
        scale[j] = temp_scale;
        zp[j] = temp_zp;
    }
    // quantize
    for (size_t i = 0; i < seq_dim; ++i) {
        for (size_t j = 0; j < hidden_dims; j++) {
            float tmp = src[i * src_stride + j];
            dst[i * dst_stride + j] = static_cast<uint8_t>(std::round(tmp / scale[j] + zp[j]));
        }
    }
}

template <typename TA, typename TB>
void cvt_copy(TA* a, TB* b, size_t m, size_t n, size_t src_stride, size_t dst_stride) {
    for (size_t j = 0; j < m; j++) {
        size_t i = 0;
#if defined(HAVE_AVX512F)
        for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
            auto vb = mm512_uni_loadu_ps(b + i + j * src_stride);
            mm512_uni_storeu_ps(a + i + j * dst_stride, vb);
        }
#elif defined(HAVE_AVX2)
        for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
            auto vb = mm256_uni_loadu_ps(b + i + j * src_stride);
            mm256_uni_storeu_ps(a + i + j * dst_stride, vb);
        }
#endif
        for (; i < n; i++) {
            a[i + j * dst_stride] = b[i + j * src_stride];
        }
    }
}

template <typename T, typename T2>
static void attn_quant_mt(const ov::intel_cpu::PlainTensor& k_src,
                          const ov::intel_cpu::PlainTensor& v_src,
                          const ov::intel_cpu::PlainTensor& k_dst,
                          const ov::intel_cpu::PlainTensor& v_dst,
                          const size_t L0,
                          float* temp_buffer,
                          const ov::intel_cpu::PlainTensor& k_scale_zp,
                          const ov::intel_cpu::PlainTensor& v_scale_zp,
                          const bool quant_key_by_channel,
                          const size_t key_group_size,
                          const size_t value_group_size) {
    // For compatibility, all input_kvs are permuted to BHLS
    size_t B = k_src.m_dims[0], H = k_src.m_dims[1], L1 = k_src.m_dims[2], S = k_src.m_dims[3], SV = v_src.m_dims[3];
    if (quant_key_by_channel) {
        if (L0 == 0) {
            parallel_for3d(ov::intel_cpu::div_up(L1, key_group_size), B, H, [&](size_t group_id, size_t b, size_t h) {
                quant_u8_by_channel_kernel(k_src.ptr<T>(b, h, group_id * key_group_size),
                                           k_dst.ptr<T2>(b, h, group_id * key_group_size),
                                           std::min(key_group_size, L1 - group_id * key_group_size),
                                           S,
                                           k_src.m_strides[2],
                                           k_dst.m_strides[2],
                                           k_scale_zp.ptr<float>(group_id * 2, b, h),
                                           k_scale_zp.ptr<float>(group_id * 2 + 1, b, h));
            });
        } else {
            size_t group_id = L0 / key_group_size;
            size_t prev_nums = L0 % key_group_size;
            parallel_for2d(B, H, [&](size_t b, size_t h) {
                auto thread_id = parallel_get_thread_num();
                float* thread_temp_buffer = temp_buffer + thread_id * key_group_size * S;
                size_t remaining_group_size = prev_nums ? (key_group_size - prev_nums) : 0;
                if (prev_nums) {
                    attn_dequant_u8_by_channel_kernel(k_dst.ptr<uint8_t>(b, h, group_id * key_group_size),
                                                      thread_temp_buffer,
                                                      prev_nums,
                                                      S,
                                                      k_dst.m_strides[2],
                                                      S,
                                                      k_scale_zp.ptr<float>(group_id * 2, b, h),
                                                      k_scale_zp.ptr<float>(group_id * 2 + 1, b, h));
                    remaining_group_size = std::min(remaining_group_size, L1);
                    cvt_copy(thread_temp_buffer + prev_nums * S,
                             k_src.ptr<T>(b, h),
                             remaining_group_size,
                             S,
                             k_src.m_strides[2],
                             S);
                    quant_u8_by_channel_kernel(thread_temp_buffer,
                                               k_dst.ptr<T2>(b, h, group_id * key_group_size),
                                               std::min(key_group_size, remaining_group_size + prev_nums),
                                               S,
                                               S,
                                               k_dst.m_strides[2],
                                               k_scale_zp.ptr<float>(group_id * 2, b, h),
                                               k_scale_zp.ptr<float>(group_id * 2 + 1, b, h));
                }

                if (L1 > remaining_group_size) {
                    size_t new_seq = L1 - remaining_group_size;
                    for (size_t new_group_id = prev_nums ? group_id + 1 : group_id, src_offset = 0;
                         new_group_id < ov::intel_cpu::div_up(L0 + L1, key_group_size);
                         new_group_id++, src_offset += key_group_size) {
                        quant_u8_by_channel_kernel(k_src.ptr<T>(b, h, remaining_group_size + src_offset),
                                                   k_dst.ptr<T2>(b, h, new_group_id * key_group_size),
                                                   std::min(key_group_size, new_seq - src_offset),
                                                   S,
                                                   k_src.m_strides[2],
                                                   k_dst.m_strides[2],
                                                   k_scale_zp.ptr<float>(new_group_id * 2, b, h),
                                                   k_scale_zp.ptr<float>(new_group_id * 2 + 1, b, h));
                    }
                }
            });
        }
    } else {
        parallel_for3d(L1, B, H, [&](size_t m, size_t b, size_t h) {
            auto p_k = k_scale_zp.ptr<float>(L0 + m, b, h);
            for (size_t group_id = 0; group_id < S / key_group_size; group_id++) {
                quant_u8(k_src.ptr<T>(b, h, m, group_id * key_group_size),
                         k_dst.ptr<T2>(b, h, L0 + m, group_id * key_group_size),
                         key_group_size,
                         p_k[group_id * 2],
                         p_k[group_id * 2 + 1]);
            }
        });
    }
    parallel_for3d(L1, B, H, [&](size_t m, size_t b, size_t h) {
        auto p_v = v_scale_zp.ptr<float>(L0 + m, b, h);
        for (size_t group_id = 0; group_id < SV / value_group_size; group_id++) {
            quant_u8(v_src.ptr<T>(b, h, m, group_id * value_group_size),
                     v_dst.ptr<T2>(b, h, L0 + m, group_id * value_group_size),
                     value_group_size,
                     p_v[group_id * 2],
                     p_v[group_id * 2 + 1]);
        }
    });
}

template <typename T, typename T2>
static void paged_attn_quant_mt(const ov::intel_cpu::PlainTensor& k_src,
                                const ov::intel_cpu::PlainTensor& v_src,
                                const ov::intel_cpu::PlainTensor& k_dst,
                                const ov::intel_cpu::PlainTensor& v_dst,
                                const ov::intel_cpu::PlainTensor& slot_mapping) {
    size_t B = k_src.m_dims[0], H = k_src.m_dims[1], L1 = k_src.m_dims[2], S = k_src.m_dims[3], SV = v_src.m_dims[3];
    size_t block_size = k_dst.m_dims[2];
    parallel_for3d(B, L1, H, [&](size_t b, size_t m, size_t h) {
        auto slot = slot_mapping.ptr<int32_t>(b)[m];
        if (slot < 0)
            return;
        auto block_number = slot / block_size;
        auto block_offset = slot % block_size;

        auto p_k = reinterpret_cast<float*>(k_dst.ptr<T2>(block_number, h, block_offset));
        auto p_v = reinterpret_cast<float*>(v_dst.ptr<T2>(block_number, h, block_offset));
        // The layout for per token per head:
        // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized
        // feature(u8,idx_S)|
        quant_u8(k_src.ptr<T>(b, h, m),
                 k_dst.ptr<T2>(block_number, h, block_offset) + sizeof(float) + sizeof(float),
                 S,
                 p_k[0],
                 p_k[1]);
        quant_u8(v_src.ptr<T>(b, h, m),
                 v_dst.ptr<T2>(block_number, h, block_offset) + sizeof(float) + sizeof(float),
                 SV,
                 p_v[0],
                 p_v[1]);
    });
}

void attn_quantkv(const ov::intel_cpu::PlainTensor& k_src,
                  const ov::intel_cpu::PlainTensor& v_src,
                  float* temp_buffer,
                  const ov::intel_cpu::PlainTensor& k_dst,
                  const ov::intel_cpu::PlainTensor& v_dst,
                  const ov::intel_cpu::PlainTensor& k_scale_zp,
                  const ov::intel_cpu::PlainTensor& v_scale_zp,
                  const size_t L0,
                  const bool quant_k_by_channel,
                  const size_t k_group_size,
                  const size_t v_group_size) {
    if (k_src.get_precision() == ov::element::f32 && k_dst.get_precision() == ov::element::u8) {
        attn_quant_mt<float, uint8_t>(k_src,
                                      v_src,
                                      k_dst,
                                      v_dst,
                                      L0,
                                      temp_buffer,
                                      k_scale_zp,
                                      v_scale_zp,
                                      quant_k_by_channel,
                                      k_group_size,
                                      v_group_size);
    } else if (k_src.get_precision() == ov::element::bf16 && k_dst.get_precision() == ov::element::u8) {
        attn_quant_mt<ov::bfloat16, uint8_t>(k_src,
                                             v_src,
                                             k_dst,
                                             v_dst,
                                             L0,
                                             temp_buffer,
                                             k_scale_zp,
                                             v_scale_zp,
                                             quant_k_by_channel,
                                             k_group_size,
                                             v_group_size);
    } else if (k_src.get_precision() == ov::element::f16 && k_dst.get_precision() == ov::element::u8) {
        attn_quant_mt<ov::float16, uint8_t>(k_src,
                                            v_src,
                                            k_dst,
                                            v_dst,
                                            L0,
                                            temp_buffer,
                                            k_scale_zp,
                                            v_scale_zp,
                                            quant_k_by_channel,
                                            k_group_size,
                                            v_group_size);
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
                        const ov::intel_cpu::PlainTensor& slot_mapping) {
    if (k_src.get_precision() == ov::element::f32 && k_dst.get_precision() == ov::element::u8) {
        paged_attn_quant_mt<float, uint8_t>(k_src, v_src, k_dst, v_dst, slot_mapping);
    } else if (k_src.get_precision() == ov::element::bf16 && k_dst.get_precision() == ov::element::u8) {
        paged_attn_quant_mt<ov::bfloat16, uint8_t>(k_src, v_src, k_dst, v_dst, slot_mapping);
    } else if (k_src.get_precision() == ov::element::f16 && k_dst.get_precision() == ov::element::u8) {
        paged_attn_quant_mt<ov::float16, uint8_t>(k_src, v_src, k_dst, v_dst, slot_mapping);
    } else {
        OPENVINO_THROW("unsupport src type: ",
                       k_src.get_precision(),
                       ", dst type: ",
                       k_dst.get_precision(),
                       " in paged_attn_quantkv");
    }
}

void attn_quant_u8(const float* src, uint8_t* dst, size_t n, float& scale, float& zp) {
    quant_u8(src, dst, n, scale, zp);
}

void attn_dequant_u8(const uint8_t* src, float* dst, size_t n, float scale, float zp) {
    attn_dequant_u8_kernel(src, dst, n, scale, zp);
}

void attn_quant_by_channel_u8(const float* src,
                              uint8_t* dst,
                              size_t seq_dim,
                              size_t hidden_dims,
                              size_t src_stride,
                              size_t dst_stride,
                              float* scale,
                              float* zp) {
    quant_u8_by_channel_kernel(src, dst, seq_dim, hidden_dims, src_stride, dst_stride, scale, zp);
}

void attn_dequant_by_channel_u8(const uint8_t* src,
                                float* dst,
                                size_t seq_dim,
                                size_t hidden_dims,
                                size_t src_stride,
                                size_t dst_stride,
                                float* scale,
                                float* zp) {
    attn_dequant_u8_by_channel_kernel(src, dst, seq_dim, hidden_dims, src_stride, dst_stride, scale, zp);
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov