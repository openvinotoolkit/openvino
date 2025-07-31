
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "nodes/kernels/scaled_attn/common.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"
#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include <cstddef>
#include <cstdint>
#if defined(HAVE_SVE)
#    include "arm_sve.h"
#endif

namespace ov::Extensions::Cpu::XARCH {

template <
    typename T,
    ov::element::Type_t SRC_PREC,
    std::enable_if_t<(std::is_same_v<T, ov::bfloat16> || std::is_same_v<T, ov::float16> || std::is_same_v<T, float>),
                     bool> = true>
void attn_acc_value_block(float* out,
                          float* weight,
                          T* v,
                          const size_t S,
                          const size_t block_size,
                          [[maybe_unused]] const size_t group_size) {
#if defined(HAVE_AVX512F)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto attn_w_vec0 = _mm512_set1_ps(weight[0]);
        auto attn_w_vec1 = _mm512_set1_ps(weight[1]);
        auto attn_w_vec2 = _mm512_set1_ps(weight[2]);
        auto attn_w_vec3 = _mm512_set1_ps(weight[3]);
        size_t i = 0;
        for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
            auto v_out = mm512_uni_loadu_ps(out + i);
            v_out = _mm512_fmadd_ps(attn_w_vec0, mm512_uni_loadu_ps(v + i), v_out);
            v_out = _mm512_fmadd_ps(attn_w_vec1, mm512_uni_loadu_ps(v + i + S), v_out);
            v_out = _mm512_fmadd_ps(attn_w_vec2, mm512_uni_loadu_ps(v + i + S * 2), v_out);
            v_out = _mm512_fmadd_ps(attn_w_vec3, mm512_uni_loadu_ps(v + i + S * 3), v_out);

            _mm512_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * v[i];
            out[i] += weight[1] * v[i + S];
            out[i] += weight[2] * v[i + S * 2];
            out[i] += weight[3] * v[i + S * 3];
        }
        v += 4 * S;
        weight += 4;
    }
    if (j + 2 <= block_size) {
        auto attn_w_vec0 = _mm512_set1_ps(weight[0]);
        auto attn_w_vec1 = _mm512_set1_ps(weight[1]);
        size_t i = 0;
        for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
            auto v_out = mm512_uni_loadu_ps(out + i);
            v_out = _mm512_fmadd_ps(attn_w_vec0, mm512_uni_loadu_ps(v + i), v_out);
            v_out = _mm512_fmadd_ps(attn_w_vec1, mm512_uni_loadu_ps(v + i + S), v_out);

            _mm512_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * v[i];
            out[i] += weight[1] * v[i + S];
        }
        v += 2 * S;
        weight += 2;
        j += 2;
    }
    if (j < block_size) {
        auto attn_w_vec0 = _mm512_set1_ps(weight[0]);
        size_t i = 0;
        for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
            auto v_out = mm512_uni_loadu_ps(out + i);
            v_out = _mm512_fmadd_ps(attn_w_vec0, mm512_uni_loadu_ps(v + i), v_out);

            _mm512_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * v[i];
        }
    }
    return;
#elif defined(HAVE_AVX2)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto attn_w_vec0 = _mm256_set1_ps(weight[0]);
        auto attn_w_vec1 = _mm256_set1_ps(weight[1]);
        auto attn_w_vec2 = _mm256_set1_ps(weight[2]);
        auto attn_w_vec3 = _mm256_set1_ps(weight[3]);
        size_t i = 0;
        for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
            auto v_out = mm256_uni_loadu_ps(out + i);
            v_out = _mm256_fmadd_ps(attn_w_vec0, mm256_uni_loadu_ps(v + i), v_out);
            v_out = _mm256_fmadd_ps(attn_w_vec1, mm256_uni_loadu_ps(v + i + S), v_out);
            v_out = _mm256_fmadd_ps(attn_w_vec2, mm256_uni_loadu_ps(v + i + S * 2), v_out);
            v_out = _mm256_fmadd_ps(attn_w_vec3, mm256_uni_loadu_ps(v + i + S * 3), v_out);

            mm256_uni_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * v[i];
            out[i] += weight[1] * v[i + S];
            out[i] += weight[2] * v[i + S * 2];
            out[i] += weight[3] * v[i + S * 3];
        }
        v += 4 * S;
        weight += 4;
    }
    if (j + 2 <= block_size) {
        auto attn_w_vec0 = _mm256_set1_ps(weight[0]);
        auto attn_w_vec1 = _mm256_set1_ps(weight[1]);
        size_t i = 0;
        for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
            auto v_out = mm256_uni_loadu_ps(out + i);
            v_out = _mm256_fmadd_ps(attn_w_vec0, mm256_uni_loadu_ps(v + i), v_out);
            v_out = _mm256_fmadd_ps(attn_w_vec1, mm256_uni_loadu_ps(v + i + S), v_out);

            mm256_uni_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * v[i];
            out[i] += weight[1] * v[i + S];
        }
        v += 2 * S;
        weight += 2;
        j += 2;
    }
    if (j < block_size) {
        auto attn_w_vec0 = _mm256_set1_ps(weight[0]);
        size_t i = 0;
        for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
            auto v_out = mm256_uni_loadu_ps(out + i);
            v_out = _mm256_fmadd_ps(attn_w_vec0, mm256_uni_loadu_ps(v + i), v_out);

            mm256_uni_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * v[i];
        }
    }
    return;
#else
    for (size_t j = 0; j < block_size; j++) {
        for (size_t i = 0; i < S; i++) {
            out[i] += weight[j] * v[i];
        }
        v += S;
    }
#endif
}
template <typename T, ov::element::Type_t SRC_PREC, std::enable_if_t<SRC_PREC == ov::element::u8, bool> = true>
void attn_acc_value_block_by_dim(float* out,
                                 float* weight,
                                 uint8_t* v,
                                 const size_t S,
                                 const size_t block_size,
                                 const size_t group_size) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized
    // feature(u8,idx_S)| The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
    size_t src_offset = 0;
    size_t dst_offset = 0;
    const size_t params_offset = sizeof(float) * 2;
    const size_t src_stride = S / group_size * (group_size + params_offset);

#if defined(HAVE_AVX512F)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        dst_offset = 0;
        src_offset = 0;
        while (dst_offset < S) {
            // process group by group
            uint8_t* v_ptr = v + src_offset;
            auto v_f0 = reinterpret_cast<float*>(v_ptr);
            auto v_f1 = reinterpret_cast<float*>(v_ptr + src_stride);
            auto v_f2 = reinterpret_cast<float*>(v_ptr + 2 * src_stride);
            auto v_f3 = reinterpret_cast<float*>(v_ptr + 3 * src_stride);
            auto attn_w_vec0 = _mm512_set1_ps(weight[0] * v_f0[0]);
            auto attn_w_vec1 = _mm512_set1_ps(weight[1] * v_f1[0]);
            auto attn_w_vec2 = _mm512_set1_ps(weight[2] * v_f2[0]);
            auto attn_w_vec3 = _mm512_set1_ps(weight[3] * v_f3[0]);
            auto zp0 = _mm512_set1_ps(v_f0[1]);
            auto zp1 = _mm512_set1_ps(v_f1[1]);
            auto zp2 = _mm512_set1_ps(v_f2[1]);
            auto zp3 = _mm512_set1_ps(v_f3[1]);
            uint8_t* v_data_ptr = v + src_offset + params_offset;
            size_t i = 0;
            for (; i + vec_len_f32_avx512 <= group_size; i += vec_len_f32_avx512) {
                auto v_out = mm512_uni_loadu_ps(out + dst_offset + i);
                auto v0 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                                            _mm_loadu_si128(reinterpret_cast<__m128i*>(v_data_ptr + i)))),
                                        zp0);
                auto v1 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                                            _mm_loadu_si128(reinterpret_cast<__m128i*>(v_data_ptr + i + src_stride)))),
                                        zp1);
                auto v2 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(
                                            reinterpret_cast<__m128i*>(v_data_ptr + i + 2 * src_stride)))),
                                        zp2);
                auto v3 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(
                                            reinterpret_cast<__m128i*>(v_data_ptr + i + 3 * src_stride)))),
                                        zp3);
                v_out = _mm512_fmadd_ps(attn_w_vec0, v0, v_out);
                v_out = _mm512_fmadd_ps(attn_w_vec1, v1, v_out);
                v_out = _mm512_fmadd_ps(attn_w_vec2, v2, v_out);
                v_out = _mm512_fmadd_ps(attn_w_vec3, v3, v_out);
                _mm512_storeu_ps(out + dst_offset + i, v_out);
            }
            for (; i < group_size; i++) {
                out[i] += weight[0] * (v_data_ptr[i] - v_f0[1]) * v_f0[0];
                out[i] += weight[1] * (v_data_ptr[i + src_stride] - v_f1[1]) * v_f1[0];
                out[i] += weight[2] * (v_data_ptr[i + 2 * src_stride] - v_f2[1]) * v_f2[0];
                out[i] += weight[3] * (v_data_ptr[i + 3 * src_stride] - v_f3[1]) * v_f3[0];
            }
            dst_offset += group_size;
            src_offset += group_size + params_offset;
        }
        weight += 4;
        v += 4 * src_stride;
    }
    for (; j < block_size; j++) {
        dst_offset = 0;
        src_offset = 0;
        while (dst_offset < S) {
            uint8_t* v_ptr = v + src_offset;
            uint8_t* v_data_ptr = v_ptr + params_offset;
            auto v_f0 = reinterpret_cast<float*>(v_ptr);
            auto attn_w_vec0 = _mm512_set1_ps(weight[0] * v_f0[0]);
            auto zp0 = _mm512_set1_ps(v_f0[1]);
            size_t i = 0;
            for (; i + vec_len_f32_avx512 <= group_size; i += vec_len_f32_avx512) {
                auto v_out = mm512_uni_loadu_ps((out + dst_offset + i));
                auto v0 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                                            _mm_loadu_si128(reinterpret_cast<__m128i*>(v_data_ptr + i)))),
                                        zp0);
                v_out = _mm512_fmadd_ps(attn_w_vec0, v0, v_out);

                _mm512_storeu_ps((out + dst_offset + i), v_out);
            }
            for (; i < group_size; i++) {
                out[dst_offset + i] += weight[0] * (v_data_ptr[i] - v_f0[1]) * v_f0[0];
            }
            dst_offset += group_size;
            src_offset += group_size + params_offset;
        }
        v += src_stride;
        weight++;
    }
    return;
#elif defined(HAVE_AVX2)
    size_t j = 0;
    for (; j < block_size; j++) {
        dst_offset = 0;
        src_offset = 0;
        while (dst_offset < S) {
            uint8_t* v_ptr = v + src_offset;
            uint8_t* v_data_ptr = v_ptr + params_offset;
            auto v_f0 = reinterpret_cast<float*>(v_ptr);
            auto attn_w_vec0 = _mm256_set1_ps(weight[0] * v_f0[0]);
            auto zp0 = _mm256_set1_ps(v_f0[1]);
            size_t i = 0;
            for (; i + vec_len_f32_avx2 <= group_size; i += vec_len_f32_avx2) {
                auto v_out = mm256_uni_loadu_ps(out + dst_offset + i);
                auto v0 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                                            _mm_loadl_epi64(reinterpret_cast<__m128i*>(v_data_ptr + i)))),
                                        zp0);
                v_out = _mm256_fmadd_ps(attn_w_vec0, v0, v_out);

                mm256_uni_storeu_ps(out + dst_offset + i, v_out);
            }
            for (; i < group_size; i++) {
                out[dst_offset + i] += weight[0] * (v_data_ptr[i] - v_f0[1]) * v_f0[0];
            }
            dst_offset += group_size;
            src_offset += group_size + params_offset;
        }
        v += src_stride;
        weight++;
    }
    return;
#elif defined(HAVE_SVE)
    auto sve_pg = svptrue_b32();
    size_t j = 0;
    for (; j < block_size; ++j) {
        dst_offset = 0;
        src_offset = 0;
        while (dst_offset < S) {
            uint8_t* v_ptr = v + src_offset;
            uint8_t* v_data_ptr = v_ptr + params_offset;
            auto v_f0 = reinterpret_cast<float*>(v_ptr);
            svfloat32_t attn_w_vec0 = svdup_n_f32(weight[0] * v_f0[0]);
            svfloat32_t zp0 = svdup_n_f32(v_f0[1]);
            size_t i = 0;
            for (; i + svcntw() <= group_size; i += svcntw()) {
                auto v_out = svld1_f32(sve_pg, out + dst_offset + i);
                svuint32_t reg1 = svld1ub_u32(sve_pg, v_data_ptr + i);
                svfloat32_t reg2 = svcvt_f32_u32_z(sve_pg, reg1);
                svfloat32_t v0 = svsub_f32_z(sve_pg, reg2, zp0);
                v_out = svmla_f32_x(sve_pg, v_out, attn_w_vec0, v0);
                svst1_f32(sve_pg, out + dst_offset + i, v_out);
            }
            auto sve_pgt = svwhilelt_b32(i, group_size);
            auto v_out = svld1_f32(sve_pgt, out + dst_offset + i);
            svuint32_t reg1 = svld1ub_u32(sve_pgt, v_data_ptr + i);
            svfloat32_t reg2 = svcvt_f32_u32_z(sve_pgt, reg1);
            svfloat32_t v0 = svsub_f32_z(sve_pgt, reg2, zp0);
            v_out = svmla_f32_x(sve_pgt, v_out, attn_w_vec0, v0);
            svst1_f32(sve_pgt, out + dst_offset + i, v_out);
            dst_offset += group_size;
            src_offset += group_size + params_offset;
        }
        v += src_stride;
        weight++;
    }
    return;
#else
    for (size_t j = 0; j < block_size; j++) {
        dst_offset = 0;
        src_offset = 0;
        while (dst_offset < S) {
            auto v0 = reinterpret_cast<float*>(v + src_offset);
            for (size_t i = 0; i < group_size; i++) {
                out[dst_offset + i] += weight[j] * (v[i + src_offset + params_offset] - v0[1]) * v0[0];
            }
            dst_offset += group_size;
            src_offset += group_size + params_offset;
        }
        v += src_stride;
    }
#endif
}

template <typename T, ov::element::Type_t SRC_PREC, std::enable_if_t<SRC_PREC == ov::element::u4, bool> = true>
void attn_acc_value_block_by_dim(float* out,
                                 float* weight,
                                 uint8_t* v_ptr,
                                 const size_t S,
                                 const size_t block_size,
                                 const size_t group_size) {
    size_t src_offset = 0;
    size_t dst_offset = 0;
    const size_t params_offset = sizeof(float) * 2;
    auto sub_byte_multiplier = 8 / 4;
    const size_t src_stride = S / group_size * (group_size / sub_byte_multiplier + params_offset);
    for (size_t j = 0; j < block_size; j++) {
        dst_offset = 0;
        src_offset = 0;
        while (dst_offset < S) {
            auto v0 = reinterpret_cast<float*>(v_ptr + src_offset);
            size_t i = 0;
#if defined(HAVE_AVX512F)
            auto attn_w_vec0 = _mm512_set1_ps(weight[j] * v0[0]);
            auto v_zp = _mm512_set1_ps(v0[1]);
            for (; i + vec_len_f32_avx512 * 2 <= group_size; i += vec_len_f32_avx512 * 2) {
                auto data = _mm_loadu_si128(reinterpret_cast<__m128i*>(v_ptr + i / 2 + src_offset + params_offset));
                auto v_i32 = _mm512_cvtepu8_epi32(data);

                auto v_512_low_half = _mm512_srli_epi32(v_i32, 4);
                auto v_f32_low_half = _mm512_cvtepi32_ps(v_512_low_half);

                auto mask = _mm512_set1_epi32(0x0F);
                auto v_512_high_half = _mm512_and_si512(v_i32, mask);
                auto v_f32_high_half = _mm512_cvtepi32_ps(v_512_high_half);

                // q - zp
                v_f32_low_half = _mm512_sub_ps(v_f32_low_half, v_zp);
                v_f32_high_half = _mm512_sub_ps(v_f32_high_half, v_zp);

                __m512i idx1 = _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
                __m512i idx2 = _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8);
                __m512 first_half = _mm512_permutex2var_ps(v_f32_low_half, idx1, v_f32_high_half);
                __m512 second_half = _mm512_permutex2var_ps(v_f32_low_half, idx2, v_f32_high_half);
                auto v_out0 = mm512_uni_loadu_ps(out + dst_offset + i);
                auto v_out1 = mm512_uni_loadu_ps(out + dst_offset + i + vec_len_f32_avx512);
                v_out0 = _mm512_fmadd_ps(attn_w_vec0, first_half, v_out0);
                v_out1 = _mm512_fmadd_ps(attn_w_vec0, second_half, v_out1);
                mm512_uni_storeu_ps(out + dst_offset + i, v_out0);
                mm512_uni_storeu_ps(out + dst_offset + i + vec_len_f32_avx512, v_out1);
            }
#elif defined(HAVE_AVX2)
            auto v256_attn_w_vec0 = _mm256_set1_ps(weight[j] * v0[0]);
            auto v256_zp = _mm256_set1_ps(v0[1]);
            for (; i + vec_len_f32_avx2 * 2 <= group_size; i += vec_len_f32_avx2 * 2) {
                auto data = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v_ptr + i / 2 + src_offset + params_offset));

                auto v_i32 = _mm256_cvtepu8_epi32(data);
                auto v_256_low_half = _mm256_srli_epi32(v_i32, 4);
                auto v_f32_low_half = _mm256_cvtepi32_ps(v_256_low_half);

                auto mask = _mm256_set1_epi32(0x0F);
                auto v_256_high_half = _mm256_and_si256(v_i32, mask);
                auto v_f32_high_half = _mm256_cvtepi32_ps(v_256_high_half);
                // q - zp
                v_f32_low_half = _mm256_sub_ps(v_f32_low_half, v256_zp);
                v_f32_high_half = _mm256_sub_ps(v_f32_high_half, v256_zp);

                auto v_out0 = mm256_uni_loadu_ps(out + dst_offset + i);
                auto v_out1 = mm256_uni_loadu_ps(out + dst_offset + i + vec_len_f32_avx2);

                __m256 first_half = _mm256_permute2f128_ps(v_f32_low_half, v_f32_high_half, 0x20);
                auto idx1 = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
                first_half = _mm256_permutevar8x32_ps(first_half, idx1);
                __m256 second_half = _mm256_permute2f128_ps(v_f32_low_half, v_f32_high_half, 0x31);
                second_half = _mm256_permutevar8x32_ps(second_half, idx1);

                v_out0 = _mm256_fmadd_ps(v256_attn_w_vec0, first_half, v_out0);
                v_out1 = _mm256_fmadd_ps(v256_attn_w_vec0, second_half, v_out1);
                mm256_uni_storeu_ps(out + dst_offset + i, v_out0);
                mm256_uni_storeu_ps(out + dst_offset + i + vec_len_f32_avx2, v_out1);
            }
#endif
            for (; i < group_size; i += 2) {
                uint8_t data = v_ptr[i / 2 + src_offset + params_offset];
                float tmp0 = extract_half_byte(data, static_cast<bool>(i % 2));
                float tmp1 = extract_half_byte(data, static_cast<bool>((i + 1) % 2));
                out[dst_offset + i] += weight[j] * (tmp0 - v0[1]) * v0[0];
                out[dst_offset + i + 1] += weight[j] * (tmp1 - v0[1]) * v0[0];
            }
            dst_offset += group_size;
            src_offset += group_size / sub_byte_multiplier + params_offset;
        }
        v_ptr += src_stride;
    }
}

template <typename T, ov::element::Type_t SRC_PREC, std::enable_if_t<SRC_PREC == ov::element::u8, bool> = true>
void attn_acc_value_block_by_channel(float* out, float* weight, void* v, const size_t S, const size_t block_size) {
    auto p_scales = reinterpret_cast<float*>(v);
    auto p_zps = p_scales + S;
    auto v_data_ptr = reinterpret_cast<uint8_t*>(v) + 2 * sizeof(float) * S;
    size_t src_stride = S;
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        size_t i = 0;
#if defined(HAVE_AVX512F)
        auto weight0 = _mm512_set1_ps(weight[j]);
        auto weight1 = _mm512_set1_ps(weight[j + 1]);
        auto weight2 = _mm512_set1_ps(weight[j + 2]);
        auto weight3 = _mm512_set1_ps(weight[j + 3]);
        for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
            auto scale = _mm512_loadu_ps(p_scales + i);
            auto zp = _mm512_loadu_ps(p_zps + i);

            auto v_out = mm512_uni_loadu_ps(out + i);
            auto v0 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                                        _mm_loadu_si128(reinterpret_cast<__m128i*>(v_data_ptr + i + j * src_stride)))),
                                    zp);
            v0 = _mm512_mul_ps(v0, scale);
            auto v1 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(
                                        reinterpret_cast<__m128i*>(v_data_ptr + i + (j + 1) * src_stride)))),
                                    zp);
            v1 = _mm512_mul_ps(v1, scale);
            auto v2 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(
                                        reinterpret_cast<__m128i*>(v_data_ptr + i + (j + 2) * src_stride)))),
                                    zp);
            v2 = _mm512_mul_ps(v2, scale);
            auto v3 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(
                                        reinterpret_cast<__m128i*>(v_data_ptr + i + (j + 3) * src_stride)))),
                                    zp);
            v3 = _mm512_mul_ps(v3, scale);
            v_out = _mm512_fmadd_ps(weight0, v0, v_out);
            v_out = _mm512_fmadd_ps(weight1, v1, v_out);
            v_out = _mm512_fmadd_ps(weight2, v2, v_out);
            v_out = _mm512_fmadd_ps(weight3, v3, v_out);
            _mm512_storeu_ps(out + i, v_out);
        }
#elif defined(HAVE_AVX2)
        auto v256_weight0 = _mm256_set1_ps(weight[j]);
        auto v256_weight1 = _mm256_set1_ps(weight[j + 1]);
        auto v256_weight2 = _mm256_set1_ps(weight[j + 2]);
        auto v256_weight3 = _mm256_set1_ps(weight[j + 3]);
        for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
            auto scale = mm256_uni_loadu_ps(p_scales + i);
            auto zp = mm256_uni_loadu_ps(p_zps + i);

            auto v_out = mm256_uni_loadu_ps(out + i);
            auto v0 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                                        _mm_loadl_epi64(reinterpret_cast<__m128i*>(v_data_ptr + i + j * src_stride)))),
                                    zp);
            v0 = _mm256_mul_ps(v0, scale);
            auto v1 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                                        reinterpret_cast<__m128i*>(v_data_ptr + i + (j + 1) * src_stride)))),
                                    zp);
            v1 = _mm256_mul_ps(v1, scale);
            auto v2 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                                        reinterpret_cast<__m128i*>(v_data_ptr + i + (j + 2) * src_stride)))),
                                    zp);
            v2 = _mm256_mul_ps(v2, scale);
            auto v3 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                                        reinterpret_cast<__m128i*>(v_data_ptr + i + (j + 3) * src_stride)))),
                                    zp);
            v3 = _mm256_mul_ps(v3, scale);
            v_out = _mm256_fmadd_ps(v256_weight0, v0, v_out);
            v_out = _mm256_fmadd_ps(v256_weight1, v1, v_out);
            v_out = _mm256_fmadd_ps(v256_weight2, v2, v_out);
            v_out = _mm256_fmadd_ps(v256_weight3, v3, v_out);
            mm256_uni_storeu_ps(out + i, v_out);
        }
#endif
        for (; i < S; i++) {
            out[i] += weight[j] * (v_data_ptr[i + j * src_stride] - p_zps[i]) * p_scales[i];
            out[i] += weight[j + 1] * (v_data_ptr[i + (j + 1) * src_stride] - p_zps[i]) * p_scales[i];
            out[i] += weight[j + 2] * (v_data_ptr[i + (j + 2) * src_stride] - p_zps[i]) * p_scales[i];
            out[i] += weight[j + 3] * (v_data_ptr[i + (j + 3) * src_stride] - p_zps[i]) * p_scales[i];
        }
    }
    for (; j < block_size; j++) {
        for (size_t i = 0; i < S; i++) {
            out[i] += weight[j] * (v_data_ptr[i + j * src_stride] - p_zps[i]) * p_scales[i];
        }
    }
}

template <typename T, ov::element::Type_t SRC_PREC, std::enable_if_t<SRC_PREC == ov::element::u4, bool> = true>
void attn_acc_value_block_by_channel(float* out, float* weight, void* v, const size_t S, const size_t block_size) {
    auto p_scales = reinterpret_cast<float*>(v);
    auto p_zps = p_scales + S;
    auto v_data_ptr = reinterpret_cast<uint8_t*>(v) + 2 * sizeof(float) * S;
    size_t src_stride = S / get_sub_byte_multiplier(SRC_PREC);
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        size_t i = 0;
#if defined(HAVE_AVX512F)
        auto weight0 = _mm512_set1_ps(weight[j]);
        auto weight1 = _mm512_set1_ps(weight[j + 1]);
        auto weight2 = _mm512_set1_ps(weight[j + 2]);
        auto weight3 = _mm512_set1_ps(weight[j + 3]);
        for (; i + vec_len_f32_avx512 * 2 <= S; i += vec_len_f32_avx512 * 2) {
            auto scale00 = _mm512_loadu_ps(p_scales + i);
            auto zp00 = _mm512_loadu_ps(p_zps + i);

            auto scale01 = _mm512_loadu_ps(p_scales + i + vec_len_f32_avx512);
            auto zp01 = _mm512_loadu_ps(p_zps + i + vec_len_f32_avx512);

            auto v_out0 = mm512_uni_loadu_ps(out + i);
            auto v_out1 = mm512_uni_loadu_ps(out + i + vec_len_f32_avx512);
            __m512 v00, v01;
            mm512_loadu_u4_to_f32(v_data_ptr + i / 2 + j * src_stride, v00, v01);
            v00 = _mm512_sub_ps(v00, zp00);
            v00 = _mm512_mul_ps(v00, scale00);
            v01 = _mm512_sub_ps(v01, zp01);
            v01 = _mm512_mul_ps(v01, scale01);

            __m512 v10, v11;
            mm512_loadu_u4_to_f32(v_data_ptr + i / 2 + (j + 1) * src_stride, v10, v11);
            v10 = _mm512_sub_ps(v10, zp00);
            v10 = _mm512_mul_ps(v10, scale00);
            v11 = _mm512_sub_ps(v11, zp01);
            v11 = _mm512_mul_ps(v11, scale01);

            __m512 v20, v21;
            mm512_loadu_u4_to_f32(v_data_ptr + i / 2 + (j + 2) * src_stride, v20, v21);
            v20 = _mm512_sub_ps(v20, zp00);
            v20 = _mm512_mul_ps(v20, scale00);
            v21 = _mm512_sub_ps(v21, zp01);
            v21 = _mm512_mul_ps(v21, scale01);

            __m512 v30, v31;
            mm512_loadu_u4_to_f32(v_data_ptr + i / 2 + (j + 3) * src_stride, v30, v31);
            v30 = _mm512_sub_ps(v30, zp00);
            v30 = _mm512_mul_ps(v30, scale00);
            v31 = _mm512_sub_ps(v31, zp01);
            v31 = _mm512_mul_ps(v31, scale01);

            v_out0 = _mm512_fmadd_ps(weight0, v00, v_out0);
            v_out1 = _mm512_fmadd_ps(weight0, v01, v_out1);

            v_out0 = _mm512_fmadd_ps(weight1, v10, v_out0);
            v_out1 = _mm512_fmadd_ps(weight1, v11, v_out1);

            v_out0 = _mm512_fmadd_ps(weight2, v20, v_out0);
            v_out1 = _mm512_fmadd_ps(weight2, v21, v_out1);

            v_out0 = _mm512_fmadd_ps(weight3, v30, v_out0);
            v_out1 = _mm512_fmadd_ps(weight3, v31, v_out1);
            _mm512_storeu_ps(out + i, v_out0);
            _mm512_storeu_ps(out + i + vec_len_f32_avx512, v_out1);
        }
#elif defined(HAVE_AVX2)
        auto v256_weight0 = _mm256_set1_ps(weight[j]);
        auto v256_weight1 = _mm256_set1_ps(weight[j + 1]);
        auto v256_weight2 = _mm256_set1_ps(weight[j + 2]);
        auto v256_weight3 = _mm256_set1_ps(weight[j + 3]);
        for (; i + vec_len_f32_avx2 * 2 <= S; i += vec_len_f32_avx2 * 2) {
            auto scale00 = _mm256_loadu_ps(p_scales + i);
            auto zp00 = _mm256_loadu_ps(p_zps + i);

            auto scale01 = _mm256_loadu_ps(p_scales + i + vec_len_f32_avx2);
            auto zp01 = _mm256_loadu_ps(p_zps + i + vec_len_f32_avx2);

            auto v_out0 = mm256_uni_loadu_ps(out + i);
            auto v_out1 = mm256_uni_loadu_ps(out + i + vec_len_f32_avx2);
            __m256 v00, v01;
            mm256_loadu_u4_to_f32(v_data_ptr + i / 2 + j * src_stride, v00, v01);
            v00 = _mm256_sub_ps(v00, zp00);
            v00 = _mm256_mul_ps(v00, scale00);
            v01 = _mm256_sub_ps(v01, zp01);
            v01 = _mm256_mul_ps(v01, scale01);

            __m256 v10, v11;
            mm256_loadu_u4_to_f32(v_data_ptr + i / 2 + (j + 1) * src_stride, v10, v11);
            v10 = _mm256_sub_ps(v10, zp00);
            v10 = _mm256_mul_ps(v10, scale00);
            v11 = _mm256_sub_ps(v11, zp01);
            v11 = _mm256_mul_ps(v11, scale01);

            __m256 v20, v21;
            mm256_loadu_u4_to_f32(v_data_ptr + i / 2 + (j + 2) * src_stride, v20, v21);
            v20 = _mm256_sub_ps(v20, zp00);
            v20 = _mm256_mul_ps(v20, scale00);
            v21 = _mm256_sub_ps(v21, zp01);
            v21 = _mm256_mul_ps(v21, scale01);

            __m256 v30, v31;
            mm256_loadu_u4_to_f32(v_data_ptr + i / 2 + (j + 3) * src_stride, v30, v31);
            v30 = _mm256_sub_ps(v30, zp00);
            v30 = _mm256_mul_ps(v30, scale00);
            v31 = _mm256_sub_ps(v31, zp01);
            v31 = _mm256_mul_ps(v31, scale01);

            v_out0 = _mm256_fmadd_ps(v256_weight0, v00, v_out0);
            v_out1 = _mm256_fmadd_ps(v256_weight0, v01, v_out1);

            v_out0 = _mm256_fmadd_ps(v256_weight1, v10, v_out0);
            v_out1 = _mm256_fmadd_ps(v256_weight1, v11, v_out1);

            v_out0 = _mm256_fmadd_ps(v256_weight2, v20, v_out0);
            v_out1 = _mm256_fmadd_ps(v256_weight2, v21, v_out1);

            v_out0 = _mm256_fmadd_ps(v256_weight3, v30, v_out0);
            v_out1 = _mm256_fmadd_ps(v256_weight3, v31, v_out1);
            _mm256_storeu_ps(out + i, v_out0);
            _mm256_storeu_ps(out + i + vec_len_f32_avx2, v_out1);
        }
#endif
        for (; i < S; i += 2) {
            uint8_t data0 = v_data_ptr[i / 2 + j * src_stride];
            float tmp00 = extract_half_byte(data0, static_cast<bool>(i % 2));
            float tmp01 = extract_half_byte(data0, static_cast<bool>((i + 1) % 2));

            out[i] += weight[j] * (tmp00 - p_zps[i]) * p_scales[i];
            out[i + 1] += weight[j] * (tmp01 - p_zps[i + 1]) * p_scales[i + 1];

            uint8_t data1 = v_data_ptr[i / 2 + (j + 1) * src_stride];
            float tmp10 = extract_half_byte(data1, static_cast<bool>(i % 2));
            float tmp11 = extract_half_byte(data1, static_cast<bool>((i + 1) % 2));

            out[i] += weight[j + 1] * (tmp10 - p_zps[i]) * p_scales[i];
            out[i + 1] += weight[j + 1] * (tmp11 - p_zps[i + 1]) * p_scales[i + 1];

            uint8_t data2 = v_data_ptr[i / 2 + (j + 2) * src_stride];
            float tmp20 = extract_half_byte(data2, static_cast<bool>(i % 2));
            float tmp21 = extract_half_byte(data2, static_cast<bool>((i + 1) % 2));

            out[i] += weight[j + 2] * (tmp20 - p_zps[i]) * p_scales[i];
            out[i + 1] += weight[j + 2] * (tmp21 - p_zps[i + 1]) * p_scales[i + 1];

            uint8_t data3 = v_data_ptr[i / 2 + (j + 3) * src_stride];
            float tmp30 = extract_half_byte(data3, static_cast<bool>(i % 2));
            float tmp31 = extract_half_byte(data3, static_cast<bool>((i + 1) % 2));

            out[i] += weight[j + 3] * (tmp30 - p_zps[i]) * p_scales[i];
            out[i + 1] += weight[j + 3] * (tmp31 - p_zps[i + 1]) * p_scales[i + 1];
        }
    }
    for (; j < block_size; j++) {
        for (size_t i = 0; i < S; i += 2) {
            uint8_t data = v_data_ptr[i / 2 + j * src_stride];
            float tmp0 = extract_half_byte(data, static_cast<bool>(i % 2));
            float tmp1 = extract_half_byte(data, static_cast<bool>((i + 1) % 2));
            out[i] += weight[j] * (tmp0 - p_zps[i]) * p_scales[i];
            out[i + 1] += weight[j] * (tmp1 - p_zps[i + 1]) * p_scales[i + 1];
        }
    }
}

template <typename TA,
          ov::element::Type_t SRC_PREC,
          std::enable_if_t<(SRC_PREC == ov::element::u8 || SRC_PREC == ov::element::u4), bool> = true>
void attn_acc_value_block_quantized(float* out,
                                    float* weight,
                                    uint8_t* v,
                                    const size_t S,
                                    const bool is_bychannel,
                                    const size_t block_size,
                                    const size_t group_size) {
    if (is_bychannel) {
        attn_acc_value_block_by_channel<TA, SRC_PREC>(out, weight, v, S, block_size);
    } else {
        attn_acc_value_block_by_dim<TA, SRC_PREC>(out, weight, v, S, block_size, group_size);
    }
}

template <typename TA,
          ov::element::Type_t SRC_PREC,
          std::enable_if_t<(ov::intel_cpu::none_of(SRC_PREC, ov::element::u4, ov::element::u8, ov::element::i8)),
                           bool> = true>
void dot_product_block(TA* a,
                       void* b,
                       float* c,
                       const size_t n,
                       const size_t block_size,
                       [[maybe_unused]] const size_t group_size) {
    auto* b_src = reinterpret_cast<typename element_type_traits<SRC_PREC>::value_type*>(b);
#if defined(HAVE_AVX512F)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto vsum0 = _mm512_setzero_ps();
        auto vsum1 = _mm512_setzero_ps();
        auto vsum2 = _mm512_setzero_ps();
        auto vsum3 = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
            auto va = mm512_uni_loadu_ps(a + i);
            vsum0 = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b_src + i), vsum0);
            vsum1 = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b_src + i + n), vsum1);
            vsum2 = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b_src + i + 2 * n), vsum2);
            vsum3 = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b_src + i + 3 * n), vsum3);
        }
        float sum0 = _mm512_reduce_add_ps(vsum0);
        float sum1 = _mm512_reduce_add_ps(vsum1);
        float sum2 = _mm512_reduce_add_ps(vsum2);
        float sum3 = _mm512_reduce_add_ps(vsum3);
        for (; i < n; i++) {
            sum0 += a[i] * b_src[i];
            sum1 += a[i] * b_src[i + n];
            sum2 += a[i] * b_src[i + 2 * n];
            sum3 += a[i] * b_src[i + 3 * n];
        }
        c[0] = sum0;
        c[1] = sum1;
        c[2] = sum2;
        c[3] = sum3;
        c += 4;
        b_src += 4 * n;
    }
    for (; j < block_size; j++) {
        auto vsum = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
            auto va = mm512_uni_loadu_ps(a + i);
            vsum = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b_src + i), vsum);
        }
        float sum = _mm512_reduce_add_ps(vsum);
        for (; i < n; i++) {
            sum += a[i] * b_src[i];
        }
        b_src += n;
        *c++ = sum;
    }
    return;
#elif defined(HAVE_AVX2)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto vsum0 = _mm256_set1_ps(0.0f);
        auto vsum1 = _mm256_set1_ps(0.0f);
        auto vsum2 = _mm256_set1_ps(0.0f);
        auto vsum3 = _mm256_set1_ps(0.0f);
        size_t i = 0;
        for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
            auto va = mm256_uni_loadu_ps(a + i);
            vsum0 = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b_src + i), vsum0);
            vsum1 = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b_src + i + n), vsum1);
            vsum2 = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b_src + i + 2 * n), vsum2);
            vsum3 = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b_src + i + 3 * n), vsum3);
        }
        hsum(vsum0);
        hsum(vsum1);
        hsum(vsum2);
        hsum(vsum3);
        float sum0 = _mm256_cvtss_f32(vsum0);
        float sum1 = _mm256_cvtss_f32(vsum1);
        float sum2 = _mm256_cvtss_f32(vsum2);
        float sum3 = _mm256_cvtss_f32(vsum3);
        for (; i < n; i++) {
            sum0 += a[i] * b_src[i];
            sum1 += a[i] * b_src[i + n];
            sum2 += a[i] * b_src[i + 2 * n];
            sum3 += a[i] * b_src[i + 3 * n];
        }
        c[0] = sum0;
        c[1] = sum1;
        c[2] = sum2;
        c[3] = sum3;
        c += 4;
        b_src += 4 * n;
    }
    for (; j < block_size; j++) {
        auto vsum = _mm256_set1_ps(0.0f);
        size_t i = 0;
        for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
            auto va = mm256_uni_loadu_ps(a + i);
            vsum = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b_src + i), vsum);
        }
        hsum(vsum);
        float sum = _mm256_cvtss_f32(vsum);
        for (; i < n; i++) {
            sum += a[i] * b_src[i];
        }
        b_src += n;
        *c++ = sum;
    }
    return;
#else
    for (size_t j = 0; j < block_size; j++) {
        float sum = 0;
        for (size_t i = 0; i < n; i++) {
            sum += a[i] * b_src[i];
        }
        b_src += n;
        *c++ = sum;
    }
#endif
}

template <typename TA, ov::element::Type_t SRC_PREC, std::enable_if_t<(SRC_PREC == ov::element::u8), bool> = true>
void dot_product_block_quantized_by_channel(TA* a, uint8_t* b, float* c, const size_t n, const size_t block_size) {
    const size_t params_offset = sizeof(float) * 2 * n;
    const size_t src_stride = n;
    auto p_scales = reinterpret_cast<float*>(b);
    auto p_zps = p_scales + n;
    for (size_t j = 0; j < block_size; j++) {
        float sum = 0.0f;
        size_t i = 0;
#if defined(HAVE_AVX512F)
        auto v512_sum0 = _mm512_set1_ps(0.0f);
        auto v512_sum1 = _mm512_set1_ps(0.0f);
        auto v512_sum2 = _mm512_set1_ps(0.0f);
        auto v512_sum3 = _mm512_set1_ps(0.0f);
        for (; i + 4 * vec_len_f32_avx512 <= n; i += vec_len_f32_avx512 * 4) {
            auto v0_zp = _mm512_loadu_ps(p_zps + i);
            auto v1_zp = _mm512_loadu_ps(p_zps + i + vec_len_f32_avx512);
            auto v2_zp = _mm512_loadu_ps(p_zps + i + vec_len_f32_avx512 * 2);
            auto v3_zp = _mm512_loadu_ps(p_zps + i + vec_len_f32_avx512 * 3);
            auto v0_scale = _mm512_loadu_ps(p_scales + i);
            auto v1_scale = _mm512_loadu_ps(p_scales + i + vec_len_f32_avx512);
            auto v2_scale = _mm512_loadu_ps(p_scales + i + vec_len_f32_avx512 * 2);
            auto v3_scale = _mm512_loadu_ps(p_scales + i + vec_len_f32_avx512 * 3);
            auto va0 = mm512_uni_loadu_ps(a + i);
            auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);
            auto va2 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 2);
            auto va3 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 3);

            auto vb0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + params_offset + i));
            auto vb1_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + params_offset + i + vec_len_f32_avx512));
            auto vb2_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + params_offset + i + vec_len_f32_avx512 * 2));
            auto vb3_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + params_offset + i + vec_len_f32_avx512 * 3));

            auto vb0_256 = _mm512_cvtepu8_epi32(vb0_128);
            auto vb1_256 = _mm512_cvtepu8_epi32(vb1_128);
            auto vb2_256 = _mm512_cvtepu8_epi32(vb2_128);
            auto vb3_256 = _mm512_cvtepu8_epi32(vb3_128);

            auto vb0 = _mm512_cvtepi32_ps(vb0_256);
            auto vb1 = _mm512_cvtepi32_ps(vb1_256);
            auto vb2 = _mm512_cvtepi32_ps(vb2_256);
            auto vb3 = _mm512_cvtepi32_ps(vb3_256);

            vb0 = _mm512_sub_ps(vb0, v0_zp);
            vb1 = _mm512_sub_ps(vb1, v1_zp);
            vb2 = _mm512_sub_ps(vb2, v2_zp);
            vb3 = _mm512_sub_ps(vb3, v3_zp);

            vb0 = _mm512_mul_ps(vb0, v0_scale);
            vb1 = _mm512_mul_ps(vb1, v1_scale);
            vb2 = _mm512_mul_ps(vb2, v2_scale);
            vb3 = _mm512_mul_ps(vb3, v3_scale);

            v512_sum0 = _mm512_fmadd_ps(va0, vb0, v512_sum0);
            v512_sum1 = _mm512_fmadd_ps(va1, vb1, v512_sum1);
            v512_sum2 = _mm512_fmadd_ps(va2, vb2, v512_sum2);
            v512_sum3 = _mm512_fmadd_ps(va3, vb3, v512_sum3);
        }
        if (i + 2 * vec_len_f32_avx512 <= n) {
            auto v0_zp = _mm512_loadu_ps(p_zps + i);
            auto v1_zp = _mm512_loadu_ps(p_zps + i + vec_len_f32_avx512);
            auto v0_scale = _mm512_loadu_ps(p_scales + i);
            auto v1_scale = _mm512_loadu_ps(p_scales + i + vec_len_f32_avx512);

            auto va0 = mm512_uni_loadu_ps(a + i);
            auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);

            auto vb0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + params_offset + i));
            auto vb1_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + params_offset + i + vec_len_f32_avx512));

            auto vb0_256 = _mm512_cvtepu8_epi32(vb0_128);
            auto vb1_256 = _mm512_cvtepu8_epi32(vb1_128);

            auto vb0 = _mm512_cvtepi32_ps(vb0_256);
            auto vb1 = _mm512_cvtepi32_ps(vb1_256);

            vb0 = _mm512_sub_ps(vb0, v0_zp);
            vb1 = _mm512_sub_ps(vb1, v1_zp);

            vb0 = _mm512_mul_ps(vb0, v0_scale);
            vb1 = _mm512_mul_ps(vb1, v1_scale);

            v512_sum0 = _mm512_fmadd_ps(va0, vb0, v512_sum0);
            v512_sum1 = _mm512_fmadd_ps(va1, vb1, v512_sum1);
            i += 2 * vec_len_f32_avx512;
        }
        if (i + vec_len_f32_avx512 <= n) {
            auto v0_zp = _mm512_loadu_ps(p_zps + i);
            auto v0_scale = _mm512_loadu_ps(p_scales + i);

            auto va0 = mm512_uni_loadu_ps(a + i);
            auto vb0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + params_offset + i));
            auto vb0_256 = _mm512_cvtepu8_epi32(vb0_128);
            auto vb0 = _mm512_cvtepi32_ps(vb0_256);
            vb0 = _mm512_sub_ps(vb0, v0_zp);
            vb0 = _mm512_mul_ps(vb0, v0_scale);
            v512_sum0 = _mm512_fmadd_ps(va0, vb0, v512_sum0);
            i += vec_len_f32_avx512;
        }
        v512_sum0 = _mm512_add_ps(v512_sum0, v512_sum1);
        v512_sum2 = _mm512_add_ps(v512_sum2, v512_sum3);
        v512_sum0 = _mm512_add_ps(v512_sum0, v512_sum2);
        sum += _mm512_reduce_add_ps(v512_sum0);
#endif
#if defined(HAVE_AVX2)
        auto vsum0 = _mm256_set1_ps(0.0f);
        auto vsum1 = _mm256_set1_ps(0.0f);
        auto vsum2 = _mm256_set1_ps(0.0f);
        auto vsum3 = _mm256_set1_ps(0.0f);
        for (; i + 4 * vec_len_f32_avx2 <= n; i += vec_len_f32_avx2 * 4) {
            auto v0_zp = _mm256_loadu_ps(p_zps + i);
            auto v1_zp = _mm256_loadu_ps(p_zps + i + vec_len_f32_avx2);
            auto v2_zp = _mm256_loadu_ps(p_zps + i + vec_len_f32_avx2 * 2);
            auto v3_zp = _mm256_loadu_ps(p_zps + i + vec_len_f32_avx2 * 3);
            auto v0_scale = _mm256_loadu_ps(p_scales + i);
            auto v1_scale = _mm256_loadu_ps(p_scales + i + vec_len_f32_avx2);
            auto v2_scale = _mm256_loadu_ps(p_scales + i + vec_len_f32_avx2 * 2);
            auto v3_scale = _mm256_loadu_ps(p_scales + i + vec_len_f32_avx2 * 3);

            auto va0 = mm256_uni_loadu_ps(a + i);
            auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);
            auto va2 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 2);
            auto va3 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 3);

            auto vb0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + params_offset + i));
            auto vb1_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + params_offset + i + vec_len_f32_avx2));
            auto vb2_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + params_offset + i + vec_len_f32_avx2 * 2));
            auto vb3_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + params_offset + i + vec_len_f32_avx2 * 3));

            auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
            auto vb1_256 = _mm256_cvtepu8_epi32(vb1_128);
            auto vb2_256 = _mm256_cvtepu8_epi32(vb2_128);
            auto vb3_256 = _mm256_cvtepu8_epi32(vb3_128);

            auto vb0 = _mm256_cvtepi32_ps(vb0_256);
            auto vb1 = _mm256_cvtepi32_ps(vb1_256);
            auto vb2 = _mm256_cvtepi32_ps(vb2_256);
            auto vb3 = _mm256_cvtepi32_ps(vb3_256);

            vb0 = _mm256_sub_ps(vb0, v0_zp);
            vb1 = _mm256_sub_ps(vb1, v1_zp);
            vb2 = _mm256_sub_ps(vb2, v2_zp);
            vb3 = _mm256_sub_ps(vb3, v3_zp);

            vb0 = _mm256_mul_ps(vb0, v0_scale);
            vb1 = _mm256_mul_ps(vb1, v1_scale);
            vb2 = _mm256_mul_ps(vb2, v2_scale);
            vb3 = _mm256_mul_ps(vb3, v3_scale);

            vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
            vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
            vsum2 = _mm256_fmadd_ps(va2, vb2, vsum2);
            vsum3 = _mm256_fmadd_ps(va3, vb3, vsum3);
        }
        if (i + 2 * vec_len_f32_avx2 <= n) {
            auto v0_zp = _mm256_loadu_ps(p_zps + i);
            auto v1_zp = _mm256_loadu_ps(p_zps + i + vec_len_f32_avx2);
            auto v0_scale = _mm256_loadu_ps(p_scales + i);
            auto v1_scale = _mm256_loadu_ps(p_scales + i + vec_len_f32_avx2);

            auto va0 = mm256_uni_loadu_ps(a + i);
            auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);

            auto vb0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + params_offset + i));
            auto vb1_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + params_offset + i + vec_len_f32_avx2));

            auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
            auto vb1_256 = _mm256_cvtepu8_epi32(vb1_128);

            auto vb0 = _mm256_cvtepi32_ps(vb0_256);
            auto vb1 = _mm256_cvtepi32_ps(vb1_256);

            vb0 = _mm256_sub_ps(vb0, v0_zp);
            vb1 = _mm256_sub_ps(vb1, v1_zp);

            vb0 = _mm256_mul_ps(vb0, v0_scale);
            vb1 = _mm256_mul_ps(vb1, v1_scale);

            vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
            vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
            i += 2 * vec_len_f32_avx2;
        }
        if (i + vec_len_f32_avx2 <= n) {
            auto v0_zp = _mm256_loadu_ps(p_zps + i);
            auto v0_scale = _mm256_loadu_ps(p_scales + i);

            auto va0 = mm256_uni_loadu_ps(a + i);
            auto vb0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + params_offset + i));
            auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
            auto vb0 = _mm256_cvtepi32_ps(vb0_256);

            vb0 = _mm256_sub_ps(vb0, v0_zp);
            vb0 = _mm256_mul_ps(vb0, v0_scale);

            vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
            i += vec_len_f32_avx2;
        }
        vsum0 = _mm256_add_ps(vsum0, vsum1);
        vsum2 = _mm256_add_ps(vsum2, vsum3);
        vsum0 = _mm256_add_ps(vsum0, vsum2);
        hsum(vsum0);
        sum += _mm256_cvtss_f32(vsum0);
#endif
        for (; i < n; i++) {
            sum += a[i] * (b[params_offset + i] - p_zps[i]) * p_scales[i];
        }
        b += src_stride;
        *c++ = sum;
    }
}

template <typename TA, ov::element::Type_t SRC_PREC, std::enable_if_t<(SRC_PREC == ov::element::u4), bool> = true>
void dot_product_block_quantized_by_channel(TA* a, uint8_t* b, float* c, const size_t n, const size_t block_size) {
    const size_t sub_byte_multiplier = 2;
    // parans scale f32 [n] + zp f32[n]
    const size_t params_offset = sizeof(float) * 2 * n;
    // src_stride must / 2 because of u4
    const size_t src_stride = n / sub_byte_multiplier;
    auto p_scales = reinterpret_cast<float*>(b);
    auto p_zps = p_scales + n;
    for (size_t j = 0; j < block_size; j++) {
        float sum = 0.0f;
        size_t i = 0;
#if defined(HAVE_AVX512F)
        auto v512_sum0 = _mm512_set1_ps(0.0f);
        auto v512_sum1 = _mm512_set1_ps(0.0f);
        __m512 vb0, vb1;
        for (; i + 2 * vec_len_f32_avx512 <= n; i += vec_len_f32_avx512 * 2) {
            auto v0_zp = _mm512_loadu_ps(p_zps + i);
            auto v1_zp = _mm512_loadu_ps(p_zps + i + vec_len_f32_avx512);
            auto v0_scale = _mm512_loadu_ps(p_scales + i);
            auto v1_scale = _mm512_loadu_ps(p_scales + i + vec_len_f32_avx512);
            auto va0 = mm512_uni_loadu_ps(a + i);
            auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);

            mm512_loadu_u4_to_f32(b + params_offset + i / 2, vb0, vb1);

            vb0 = _mm512_sub_ps(vb0, v0_zp);
            vb1 = _mm512_sub_ps(vb1, v1_zp);

            vb0 = _mm512_mul_ps(vb0, v0_scale);
            vb1 = _mm512_mul_ps(vb1, v1_scale);

            v512_sum0 = _mm512_fmadd_ps(va0, vb0, v512_sum0);
            v512_sum1 = _mm512_fmadd_ps(va1, vb1, v512_sum1);
        }
        v512_sum0 = _mm512_add_ps(v512_sum0, v512_sum1);
        sum += _mm512_reduce_add_ps(v512_sum0);
#endif
#if defined(HAVE_AVX2)
        auto vsum0 = _mm256_set1_ps(0.0f);
        auto vsum1 = _mm256_set1_ps(0.0f);
        __m256 v256_b0, v256_b1;
        for (; i + 2 * vec_len_f32_avx2 <= n; i += vec_len_f32_avx2 * 2) {
            auto v0_zp = _mm256_loadu_ps(p_zps + i);
            auto v1_zp = _mm256_loadu_ps(p_zps + i + vec_len_f32_avx2);
            auto v0_scale = _mm256_loadu_ps(p_scales + i);
            auto v1_scale = _mm256_loadu_ps(p_scales + i + vec_len_f32_avx2);

            auto va0 = mm256_uni_loadu_ps(a + i);
            auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);

            mm256_loadu_u4_to_f32(b + params_offset + i / 2, v256_b0, v256_b1);

            v256_b0 = _mm256_sub_ps(v256_b0, v0_zp);
            v256_b1 = _mm256_sub_ps(v256_b1, v1_zp);

            v256_b0 = _mm256_mul_ps(v256_b0, v0_scale);
            v256_b1 = _mm256_mul_ps(v256_b1, v1_scale);

            vsum0 = _mm256_fmadd_ps(va0, v256_b0, vsum0);
            vsum1 = _mm256_fmadd_ps(va1, v256_b1, vsum1);
        }
        vsum0 = _mm256_add_ps(vsum0, vsum1);
        hsum(vsum0);
        sum += _mm256_cvtss_f32(vsum0);
#endif
        for (; i < n; i += 2) {
            uint8_t data = b[i / 2 + params_offset];
            float tmp0 = extract_half_byte(data, static_cast<bool>(i % 2));
            float tmp1 = extract_half_byte(data, static_cast<bool>((i + 1) % 2));
            sum += a[i] * (tmp0 - p_zps[i]) * p_scales[i];
            sum += a[i + 1] * (tmp1 - p_zps[i + 1]) * p_scales[i + 1];
        }
        b += src_stride;
        *c++ = sum;
    }
}

template <typename TA, ov::element::Type_t SRC_PREC, std::enable_if_t<(SRC_PREC == ov::element::i8), bool> = true>
void dot_product_block_quantized_by_dims(TA* a,
                                         void* b,
                                         float* c,
                                         const size_t n,
                                         const size_t block_size,
                                         const size_t group_size) {
    // The layout for per token per head:
    // |scale(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized
    // feature(u8,idx_S)| The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
    size_t src_offset = 0;
    size_t dst_offset = 0;
    const size_t params_offset = sizeof(float) * 1;
    constexpr size_t sub_byte_multiplier = get_sub_byte_multiplier(SRC_PREC);
    auto* b_src = reinterpret_cast<int8_t*>(b);
    const size_t src_stride = n / group_size * (group_size / sub_byte_multiplier + params_offset);
#if defined(HAVE_AVX512F)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        src_offset = 0;
        dst_offset = 0;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        while (dst_offset < n) {
            auto vsum0 = _mm512_setzero_ps();
            auto vsum1 = _mm512_setzero_ps();
            auto vsum2 = _mm512_setzero_ps();
            auto vsum3 = _mm512_setzero_ps();
            auto b0 = reinterpret_cast<float*>(b_src + src_offset);
            auto b1 = reinterpret_cast<float*>(b_src + src_offset + src_stride);
            auto b2 = reinterpret_cast<float*>(b_src + src_offset + src_stride * 2);
            auto b3 = reinterpret_cast<float*>(b_src + src_offset + src_stride * 3);
            size_t i = 0;
            int8_t* b_data_ptr = b_src + src_offset + params_offset;
            for (; i + vec_len_f32_avx512 <= group_size; i += vec_len_f32_avx512) {
                auto va = mm512_uni_loadu_ps(a + dst_offset + i);
                auto vb0 = _mm512_cvtepi32_ps(
                    _mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b_data_ptr + i))));
                auto vb1 = _mm512_cvtepi32_ps(
                    _mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b_data_ptr + i + src_stride))));
                auto vb2 = _mm512_cvtepi32_ps(
                    _mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b_data_ptr + i + 2 * src_stride))));
                auto vb3 = _mm512_cvtepi32_ps(
                    _mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b_data_ptr + i + 3 * src_stride))));

                vsum0 = _mm512_fmadd_ps(va, vb0, vsum0);
                vsum1 = _mm512_fmadd_ps(va, vb1, vsum1);
                vsum2 = _mm512_fmadd_ps(va, vb2, vsum2);
                vsum3 = _mm512_fmadd_ps(va, vb3, vsum3);
            }
            float group_sum0 = _mm512_reduce_add_ps(vsum0);
            float group_sum1 = _mm512_reduce_add_ps(vsum1);
            float group_sum2 = _mm512_reduce_add_ps(vsum2);
            float group_sum3 = _mm512_reduce_add_ps(vsum3);
            for (; i < group_size; i++) {
                group_sum0 += a[i + dst_offset] * (b_data_ptr[i]);
                group_sum1 += a[i + dst_offset] * (b_data_ptr[i + src_stride]);
                group_sum2 += a[i + dst_offset] * (b_data_ptr[i + 2 * src_stride]);
                group_sum3 += a[i + dst_offset] * (b_data_ptr[i + 3 * src_stride]);
            }
            sum0 += group_sum0 * b0[0];
            sum1 += group_sum1 * b1[0];
            sum2 += group_sum2 * b2[0];
            sum3 += group_sum3 * b3[0];
            dst_offset += group_size;
            src_offset += group_size + params_offset;
        }
        c[0] = sum0;
        c[1] = sum1;
        c[2] = sum2;
        c[3] = sum3;
        c += 4;
        b_src += 4 * src_stride;
    }
    for (; j < block_size; j++) {
        src_offset = 0;
        dst_offset = 0;
        float sum = 0;
        while (dst_offset < n) {
            auto vsum = _mm512_setzero_ps();
            auto b0 = reinterpret_cast<float*>(b_src + src_offset);
            size_t i = 0;
            int8_t* b_data_ptr = b_src + src_offset + params_offset;
            for (; i + vec_len_f32_avx512 <= group_size; i += vec_len_f32_avx512) {
                auto va = mm512_uni_loadu_ps(a + dst_offset + i);
                auto vb = _mm512_cvtepi32_ps(
                    _mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b_data_ptr + i))));
                vsum = _mm512_fmadd_ps(va, vb, vsum);
            }
            float group_sum = _mm512_reduce_add_ps(vsum);
            for (; i < group_size; i++) {
                group_sum += a[i + dst_offset] * (b_data_ptr[i]);
            }
            sum += group_sum * b0[0];
            dst_offset += group_size;
            src_offset += group_size + params_offset;
        }
        b_src += src_stride;
        *c++ = sum;
    }
    return;
#elif defined(HAVE_AVX2)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        src_offset = 0;
        dst_offset = 0;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        while (dst_offset < n) {
            auto vsum0 = _mm256_setzero_ps();
            auto vsum1 = _mm256_setzero_ps();
            auto vsum2 = _mm256_setzero_ps();
            auto vsum3 = _mm256_setzero_ps();
            auto b0 = reinterpret_cast<float*>(b_src + src_offset);
            auto b1 = reinterpret_cast<float*>(b_src + src_offset + src_stride);
            auto b2 = reinterpret_cast<float*>(b_src + src_offset + src_stride * 2);
            auto b3 = reinterpret_cast<float*>(b_src + src_offset + src_stride * 3);
            size_t i = 0;
            int8_t* b_data_ptr = b_src + src_offset + params_offset;
            for (; i + vec_len_f32_avx2 <= group_size; i += vec_len_f32_avx2) {
                auto va = mm256_uni_loadu_ps(a + dst_offset + i);
                auto vb0 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<__m128i*>(b_data_ptr + i))));
                auto vb1 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<__m128i*>(b_data_ptr + i + src_stride))));
                auto vb2 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<__m128i*>(b_data_ptr + i + 2 * src_stride))));
                auto vb3 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<__m128i*>(b_data_ptr + i + 3 * src_stride))));

                vsum0 = _mm256_fmadd_ps(va, vb0, vsum0);
                vsum1 = _mm256_fmadd_ps(va, vb1, vsum1);
                vsum2 = _mm256_fmadd_ps(va, vb2, vsum2);
                vsum3 = _mm256_fmadd_ps(va, vb3, vsum3);
            }
            hsum(vsum0);
            hsum(vsum1);
            hsum(vsum2);
            hsum(vsum3);
            float group_sum0 = _mm256_cvtss_f32(vsum0);
            float group_sum1 = _mm256_cvtss_f32(vsum1);
            float group_sum2 = _mm256_cvtss_f32(vsum2);
            float group_sum3 = _mm256_cvtss_f32(vsum3);
            for (; i < group_size; i++) {
                group_sum0 += a[dst_offset + i] * (b_src[i]);
                group_sum1 += a[dst_offset + i] * (b_src[i + src_stride]);
                group_sum2 += a[dst_offset + i] * (b_src[i + 2 * src_stride]);
                group_sum3 += a[dst_offset + i] * (b_src[i + 3 * src_stride]);
            }
            sum0 += group_sum0 * b0[0];
            sum1 += group_sum1 * b1[0];
            sum2 += group_sum2 * b2[0];
            sum3 += group_sum3 * b3[0];
            dst_offset += group_size;
            src_offset += group_size + params_offset;
        }
        c[0] = sum0;
        c[1] = sum1;
        c[2] = sum2;
        c[3] = sum3;
        c += 4;
        b_src += 4 * src_stride;
    }
    for (; j < block_size; j++) {
        src_offset = 0;
        dst_offset = 0;
        float sum = 0;
        while (dst_offset < n) {
            auto vsum = _mm256_setzero_ps();
            auto b0 = reinterpret_cast<float*>(b_src + src_offset);
            size_t i = 0;
            int8_t* b_data_ptr = b_src + src_offset + params_offset;
            for (; i + vec_len_f32_avx2 <= group_size; i += vec_len_f32_avx2) {
                auto va = mm256_uni_loadu_ps(a + dst_offset + i);
                auto vb = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<__m128i*>(b_data_ptr + i))));
                vsum = _mm256_fmadd_ps(va, vb, vsum);
            }
            hsum(vsum);
            float group_sum = _mm256_cvtss_f32(vsum);
            for (; i < group_size; i++) {
                group_sum += a[i + dst_offset] * (b_data_ptr[i]);
            }
            sum += group_sum * b0[0];
            dst_offset += group_size;
            src_offset += group_size + params_offset;
        }
        b_src += src_stride;
        *c++ = sum;
    }
    return;
#else
    for (size_t j = 0; j < block_size; j++) {
        float sum = 0;
        dst_offset = 0;
        src_offset = 0;
        while (dst_offset < n) {
            auto b0 = reinterpret_cast<float*>(b_src + src_offset);
            float group_sum = 0.0f;
            for (size_t i = 0; i < group_size; i++) {
                group_sum += a[dst_offset + i] * (b_src[src_offset + params_offset + i]);
            }
            sum += group_sum * b0[0];
            dst_offset += group_size;
            src_offset += group_size + params_offset;
        }
        b_src += src_stride;
        *c++ = sum;
    }
#endif
}

template <typename TA, ov::element::Type_t SRC_PREC, std::enable_if_t<(SRC_PREC == ov::element::u8), bool> = true>
void dot_product_block_quantized_by_dims(TA* a,
                                         uint8_t* b,
                                         float* c,
                                         const size_t n,
                                         const size_t block_size,
                                         const size_t group_size) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized
    // feature(u8,idx_S)| The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
    size_t src_offset = 0;
    size_t dst_offset = 0;
    const size_t params_offset = sizeof(float) * 2;
    constexpr size_t sub_byte_multiplier = get_sub_byte_multiplier(SRC_PREC);
    const size_t src_stride = n / group_size * (group_size / sub_byte_multiplier + params_offset);
#if defined(HAVE_AVX512F)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        src_offset = 0;
        dst_offset = 0;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        while (dst_offset < n) {
            auto vsum0 = _mm512_setzero_ps();
            auto vsum1 = _mm512_setzero_ps();
            auto vsum2 = _mm512_setzero_ps();
            auto vsum3 = _mm512_setzero_ps();
            auto b0 = reinterpret_cast<float*>(b + src_offset);
            auto b1 = reinterpret_cast<float*>(b + src_offset + src_stride);
            auto b2 = reinterpret_cast<float*>(b + src_offset + src_stride * 2);
            auto b3 = reinterpret_cast<float*>(b + src_offset + src_stride * 3);
            auto v_zp0 = _mm512_set1_ps(b0[1]);
            auto v_zp1 = _mm512_set1_ps(b1[1]);
            auto v_zp2 = _mm512_set1_ps(b2[1]);
            auto v_zp3 = _mm512_set1_ps(b3[1]);
            size_t i = 0;
            uint8_t* b_data_ptr = b + src_offset + params_offset;
            for (; i + vec_len_f32_avx512 <= group_size; i += vec_len_f32_avx512) {
                auto va = mm512_uni_loadu_ps(a + dst_offset + i);
                auto vb0 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                                             _mm_loadu_si128(reinterpret_cast<__m128i*>(b_data_ptr + i)))),
                                         v_zp0);
                auto vb1 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                                             _mm_loadu_si128(reinterpret_cast<__m128i*>(b_data_ptr + i + src_stride)))),
                                         v_zp1);
                auto vb2 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(
                                             reinterpret_cast<__m128i*>(b_data_ptr + i + 2 * src_stride)))),
                                         v_zp2);
                auto vb3 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(
                                             reinterpret_cast<__m128i*>(b_data_ptr + i + 3 * src_stride)))),
                                         v_zp3);

                vsum0 = _mm512_fmadd_ps(va, vb0, vsum0);
                vsum1 = _mm512_fmadd_ps(va, vb1, vsum1);
                vsum2 = _mm512_fmadd_ps(va, vb2, vsum2);
                vsum3 = _mm512_fmadd_ps(va, vb3, vsum3);
            }
            float group_sum0 = _mm512_reduce_add_ps(vsum0);
            float group_sum1 = _mm512_reduce_add_ps(vsum1);
            float group_sum2 = _mm512_reduce_add_ps(vsum2);
            float group_sum3 = _mm512_reduce_add_ps(vsum3);
            for (; i < group_size; i++) {
                group_sum0 += a[i + dst_offset] * (b_data_ptr[i] - b0[1]);
                group_sum1 += a[i + dst_offset] * (b_data_ptr[i + src_stride] - b1[1]);
                group_sum2 += a[i + dst_offset] * (b_data_ptr[i + 2 * src_stride] - b2[1]);
                group_sum3 += a[i + dst_offset] * (b_data_ptr[i + 3 * src_stride] - b3[1]);
            }
            sum0 += group_sum0 * b0[0];
            sum1 += group_sum1 * b1[0];
            sum2 += group_sum2 * b2[0];
            sum3 += group_sum3 * b3[0];
            dst_offset += group_size;
            src_offset += group_size + params_offset;
        }
        c[0] = sum0;
        c[1] = sum1;
        c[2] = sum2;
        c[3] = sum3;
        c += 4;
        b += 4 * src_stride;
    }
    for (; j < block_size; j++) {
        src_offset = 0;
        dst_offset = 0;
        float sum = 0;
        while (dst_offset < n) {
            auto vsum = _mm512_setzero_ps();
            auto b0 = reinterpret_cast<float*>(b + src_offset);
            auto v_zp = _mm512_set1_ps(b0[1]);
            size_t i = 0;
            uint8_t* b_data_ptr = b + src_offset + params_offset;
            for (; i + vec_len_f32_avx512 <= group_size; i += vec_len_f32_avx512) {
                auto va = mm512_uni_loadu_ps(a + dst_offset + i);
                auto vb = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                                            _mm_loadu_si128(reinterpret_cast<__m128i*>(b_data_ptr + i)))),
                                        v_zp);
                vsum = _mm512_fmadd_ps(va, vb, vsum);
            }
            float group_sum = _mm512_reduce_add_ps(vsum);
            for (; i < group_size; i++) {
                group_sum += a[i + dst_offset] * (b_data_ptr[i] - b0[1]);
            }
            sum += group_sum * b0[0];
            dst_offset += group_size;
            src_offset += group_size + params_offset;
        }
        b += src_stride;
        *c++ = sum;
    }
    return;
#elif defined(HAVE_AVX2)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        src_offset = 0;
        dst_offset = 0;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        while (dst_offset < n) {
            auto vsum0 = _mm256_setzero_ps();
            auto vsum1 = _mm256_setzero_ps();
            auto vsum2 = _mm256_setzero_ps();
            auto vsum3 = _mm256_setzero_ps();
            auto b0 = reinterpret_cast<float*>(b + src_offset);
            auto b1 = reinterpret_cast<float*>(b + src_offset + src_stride);
            auto b2 = reinterpret_cast<float*>(b + src_offset + src_stride * 2);
            auto b3 = reinterpret_cast<float*>(b + src_offset + src_stride * 3);
            auto v_zp0 = _mm256_set1_ps(b0[1]);
            auto v_zp1 = _mm256_set1_ps(b1[1]);
            auto v_zp2 = _mm256_set1_ps(b2[1]);
            auto v_zp3 = _mm256_set1_ps(b3[1]);
            size_t i = 0;
            uint8_t* b_data_ptr = b + src_offset + params_offset;
            for (; i + vec_len_f32_avx2 <= group_size; i += vec_len_f32_avx2) {
                auto va = mm256_uni_loadu_ps(a + dst_offset + i);
                auto vb0 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                                             _mm_loadl_epi64(reinterpret_cast<__m128i*>(b_data_ptr + i)))),
                                         v_zp0);
                auto vb1 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                                             _mm_loadl_epi64(reinterpret_cast<__m128i*>(b_data_ptr + i + src_stride)))),
                                         v_zp1);
                auto vb2 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                                             reinterpret_cast<__m128i*>(b_data_ptr + i + 2 * src_stride)))),
                                         v_zp2);
                auto vb3 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                                             reinterpret_cast<__m128i*>(b_data_ptr + i + 3 * src_stride)))),
                                         v_zp3);

                vsum0 = _mm256_fmadd_ps(va, vb0, vsum0);
                vsum1 = _mm256_fmadd_ps(va, vb1, vsum1);
                vsum2 = _mm256_fmadd_ps(va, vb2, vsum2);
                vsum3 = _mm256_fmadd_ps(va, vb3, vsum3);
            }
            hsum(vsum0);
            hsum(vsum1);
            hsum(vsum2);
            hsum(vsum3);
            float group_sum0 = _mm256_cvtss_f32(vsum0);
            float group_sum1 = _mm256_cvtss_f32(vsum1);
            float group_sum2 = _mm256_cvtss_f32(vsum2);
            float group_sum3 = _mm256_cvtss_f32(vsum3);
            for (; i < group_size; i++) {
                group_sum0 += a[dst_offset + i] * (b[i] - b0[1]);
                group_sum1 += a[dst_offset + i] * (b[i + src_stride] - b1[1]);
                group_sum2 += a[dst_offset + i] * (b[i + 2 * src_stride] - b2[1]);
                group_sum3 += a[dst_offset + i] * (b[i + 3 * src_stride] - b3[1]);
            }
            sum0 += group_sum0 * b0[0];
            sum1 += group_sum1 * b1[0];
            sum2 += group_sum2 * b2[0];
            sum3 += group_sum3 * b3[0];
            dst_offset += group_size;
            src_offset += group_size + params_offset;
        }
        c[0] = sum0;
        c[1] = sum1;
        c[2] = sum2;
        c[3] = sum3;
        c += 4;
        b += 4 * src_stride;
    }
    for (; j < block_size; j++) {
        src_offset = 0;
        dst_offset = 0;
        float sum = 0;
        while (dst_offset < n) {
            auto vsum = _mm256_setzero_ps();
            auto b0 = reinterpret_cast<float*>(b + src_offset);
            auto v_zp = _mm256_set1_ps(b0[1]);
            size_t i = 0;
            uint8_t* b_data_ptr = b + src_offset + params_offset;
            for (; i + vec_len_f32_avx2 <= group_size; i += vec_len_f32_avx2) {
                auto va = mm256_uni_loadu_ps(a + dst_offset + i);
                auto vb = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                                            _mm_loadl_epi64(reinterpret_cast<__m128i*>(b_data_ptr + i)))),
                                        v_zp);
                vsum = _mm256_fmadd_ps(va, vb, vsum);
            }
            hsum(vsum);
            float group_sum = _mm256_cvtss_f32(vsum);
            for (; i < group_size; i++) {
                group_sum += a[i + dst_offset] * (b_data_ptr[i] - b0[1]);
            }
            sum += group_sum * b0[0];
            dst_offset += group_size;
            src_offset += group_size + params_offset;
        }
        b += src_stride;
        *c++ = sum;
    }
    return;
#elif defined(HAVE_SVE)
    auto scratch = svdup_f16_x(svptrue_b16(), 0);
    size_t j = 0;
    for (; j < block_size; j++) {
        src_offset = 0;
        dst_offset = 0;
        float sum = 0;
        while (dst_offset < n) {
            auto vsum = svdup_n_f32(0.0f);
            auto b0 = reinterpret_cast<float*>(b + src_offset);
            auto v_zp = svdup_n_f32(b0[1]);
            size_t i = 0;
            uint8_t* b_data_ptr = b + src_offset + params_offset;
            auto vlen = svcntw();
            svbool_t pg_a, pg_f16;
            for (; i <= group_size; i += vlen) {
                svfloat32_t va;
                pg_a = svwhilelt_b32(i, group_size);
                pg_f16 = svand_z(svptrue_b16(), svwhilelt_b16(svcnth() / 2, svcnth()), svwhilelt_b16(i, group_size));
                if constexpr (std::is_same<TA, ov::float16>::value) {
                    auto load_src = svld1_f16(pg_f16, reinterpret_cast<float16_t*>(a + dst_offset + i));
                    auto src_interleave = svzip1_f16(load_src, scratch);
                    va = svcvt_f32_f16_z(pg_a, src_interleave);
                } else {
                    va = svld1(pg_a, a + dst_offset + i);
                }
                auto r1 = svcvt_f32_u32_z(pg_a, svld1ub_u32(pg_a, b_data_ptr + i));
                auto vb = svsub_f32_z(pg_a, r1, v_zp);
                vsum = svmla_f32_m(pg_a, vsum, va, vb);
            }
            float group_sum = svaddv_f32(svptrue_b32(), vsum);
            sum += group_sum * b0[0];
            dst_offset += group_size;
            src_offset += group_size + params_offset;
        }
        b += src_stride;
        *c++ = sum;
    }
    return;
#else
    for (size_t j = 0; j < block_size; j++) {
        float sum = 0;
        dst_offset = 0;
        src_offset = 0;
        while (dst_offset < n) {
            auto b0 = reinterpret_cast<float*>(b + src_offset);
            float group_sum = 0.0f;
            for (size_t i = 0; i < group_size; i++) {
                group_sum += a[dst_offset + i] * (b[src_offset + params_offset + i] - b0[1]);
            }
            sum += group_sum * b0[0];
            dst_offset += group_size;
            src_offset += group_size + params_offset;
        }
        b += src_stride;
        *c++ = sum;
    }
#endif
}

template <typename TA, ov::element::Type_t SRC_PREC, std::enable_if_t<(SRC_PREC == ov::element::u4), bool> = true>
void dot_product_block_quantized_by_dims(TA* a,
                                         uint8_t* b,
                                         float* c,
                                         const size_t n,
                                         const size_t block_size,
                                         const size_t group_size) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized
    // feature(u8,idx_S)| The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
    size_t src_offset = 0;
    size_t dst_offset = 0;
    const size_t params_offset = sizeof(float) * 2;
    const size_t sub_byte_multiplier = 2;
    const size_t src_stride = n / group_size * (group_size / sub_byte_multiplier + params_offset);
    size_t j = 0;
#if defined(HAVE_AVX512F)
    for (; j + 4 <= block_size; j += 4) {
        src_offset = 0;
        dst_offset = 0;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        while (dst_offset < n) {
            auto vsum0 = _mm512_setzero_ps();
            auto vsum1 = _mm512_setzero_ps();
            auto vsum2 = _mm512_setzero_ps();
            auto vsum3 = _mm512_setzero_ps();
            auto b0 = reinterpret_cast<float*>(b + src_offset);
            auto b1 = reinterpret_cast<float*>(b + src_offset + src_stride);
            auto b2 = reinterpret_cast<float*>(b + src_offset + src_stride * 2);
            auto b3 = reinterpret_cast<float*>(b + src_offset + src_stride * 3);
            auto v_zp0 = _mm512_set1_ps(b0[1]);
            auto v_zp1 = _mm512_set1_ps(b1[1]);
            auto v_zp2 = _mm512_set1_ps(b2[1]);
            auto v_zp3 = _mm512_set1_ps(b3[1]);
            size_t i = 0;
            uint8_t* b_data_ptr = b + src_offset + params_offset;
            for (; i + vec_len_f32_avx512 * 2 <= group_size; i += vec_len_f32_avx512 * 2) {
                auto va0 = mm512_uni_loadu_ps(a + dst_offset + i);
                auto va1 = mm512_uni_loadu_ps(a + dst_offset + i + vec_len_f32_avx512);
                __m512 vb00, vb01;
                mm512_loadu_u4_to_f32(b_data_ptr + i / 2, vb00, vb01);
                vb00 = _mm512_sub_ps(vb00, v_zp0);
                vb01 = _mm512_sub_ps(vb01, v_zp0);
                __m512 vb10, vb11;
                mm512_loadu_u4_to_f32(b_data_ptr + i / 2 + src_stride, vb10, vb11);
                vb10 = _mm512_sub_ps(vb10, v_zp1);
                vb11 = _mm512_sub_ps(vb11, v_zp1);
                __m512 vb20, vb21;
                mm512_loadu_u4_to_f32(b_data_ptr + i / 2 + 2 * src_stride, vb20, vb21);
                vb20 = _mm512_sub_ps(vb20, v_zp2);
                vb21 = _mm512_sub_ps(vb21, v_zp2);
                __m512 vb30, vb31;
                mm512_loadu_u4_to_f32(b_data_ptr + i / 2 + 3 * src_stride, vb30, vb31);
                vb30 = _mm512_sub_ps(vb30, v_zp3);
                vb31 = _mm512_sub_ps(vb31, v_zp3);

                vsum0 = _mm512_fmadd_ps(va0, vb00, vsum0);
                vsum0 = _mm512_fmadd_ps(va1, vb01, vsum0);

                vsum1 = _mm512_fmadd_ps(va0, vb10, vsum1);
                vsum1 = _mm512_fmadd_ps(va1, vb11, vsum1);

                vsum2 = _mm512_fmadd_ps(va0, vb20, vsum2);
                vsum2 = _mm512_fmadd_ps(va1, vb21, vsum2);

                vsum3 = _mm512_fmadd_ps(va0, vb30, vsum3);
                vsum3 = _mm512_fmadd_ps(va1, vb31, vsum3);
            }
            float group_sum0 = _mm512_reduce_add_ps(vsum0);
            float group_sum1 = _mm512_reduce_add_ps(vsum1);
            float group_sum2 = _mm512_reduce_add_ps(vsum2);
            float group_sum3 = _mm512_reduce_add_ps(vsum3);

            for (; i < group_size; i += 2) {
                uint8_t data = b_data_ptr[i / 2];
                float tmp0 = extract_half_byte(data, static_cast<bool>(i % 2));
                float tmp1 = extract_half_byte(data, static_cast<bool>((i + 1) % 2));
                group_sum0 += a[dst_offset + i] * (tmp0 - b0[1]);
                group_sum0 += a[dst_offset + i + 1] * (tmp1 - b0[1]);

                uint8_t data1 = b_data_ptr[i / 2 + src_stride];
                float tmp10 = extract_half_byte(data1, static_cast<bool>(i % 2));
                float tmp11 = extract_half_byte(data1, static_cast<bool>((i + 1) % 2));
                group_sum1 += a[dst_offset + i] * (tmp10 - b1[1]);
                group_sum1 += a[dst_offset + i + 1] * (tmp11 - b1[1]);

                uint8_t data2 = b_data_ptr[i / 2 + 2 * src_stride];
                float tmp20 = extract_half_byte(data2, static_cast<bool>(i % 2));
                float tmp21 = extract_half_byte(data2, static_cast<bool>((i + 1) % 2));
                group_sum2 += a[dst_offset + i] * (tmp20 - b2[1]);
                group_sum2 += a[dst_offset + i + 1] * (tmp21 - b2[1]);

                uint8_t data3 = b_data_ptr[i / 2 + 3 * src_stride];
                float tmp30 = extract_half_byte(data3, static_cast<bool>(i % 2));
                float tmp31 = extract_half_byte(data3, static_cast<bool>((i + 1) % 2));
                group_sum3 += a[dst_offset + i] * (tmp30 - b3[1]);
                group_sum3 += a[dst_offset + i + 1] * (tmp31 - b3[1]);
            }
            sum0 += group_sum0 * b0[0];
            sum1 += group_sum1 * b1[0];
            sum2 += group_sum2 * b2[0];
            sum3 += group_sum3 * b3[0];
            dst_offset += group_size;
            src_offset += group_size / sub_byte_multiplier + params_offset;
        }
        c[0] = sum0;
        c[1] = sum1;
        c[2] = sum2;
        c[3] = sum3;
        c += 4;
        b += 4 * src_stride;
    }
#elif defined(HAVE_AVX2)
    for (; j + 4 <= block_size; j += 4) {
        src_offset = 0;
        dst_offset = 0;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        while (dst_offset < n) {
            auto vsum0 = _mm256_setzero_ps();
            auto vsum1 = _mm256_setzero_ps();
            auto vsum2 = _mm256_setzero_ps();
            auto vsum3 = _mm256_setzero_ps();
            auto b0 = reinterpret_cast<float*>(b + src_offset);
            auto b1 = reinterpret_cast<float*>(b + src_offset + src_stride);
            auto b2 = reinterpret_cast<float*>(b + src_offset + src_stride * 2);
            auto b3 = reinterpret_cast<float*>(b + src_offset + src_stride * 3);
            auto v_zp0 = _mm256_set1_ps(b0[1]);
            auto v_zp1 = _mm256_set1_ps(b1[1]);
            auto v_zp2 = _mm256_set1_ps(b2[1]);
            auto v_zp3 = _mm256_set1_ps(b3[1]);
            size_t i = 0;
            uint8_t* b_data_ptr = b + src_offset + params_offset;
            for (; i + vec_len_f32_avx2 <= group_size; i += vec_len_f32_avx2 * 2) {
                auto va0 = mm256_uni_loadu_ps(a + dst_offset + i);
                auto va1 = mm256_uni_loadu_ps(a + dst_offset + i + vec_len_f32_avx2);

                __m256 vb00, vb01;
                mm256_loadu_u4_to_f32(b_data_ptr + i / 2, vb00, vb01);
                vb00 = _mm256_sub_ps(vb00, v_zp0);
                vb01 = _mm256_sub_ps(vb01, v_zp0);

                __m256 vb10, vb11;
                mm256_loadu_u4_to_f32(b_data_ptr + i / 2 + src_stride, vb10, vb11);
                vb10 = _mm256_sub_ps(vb10, v_zp1);
                vb11 = _mm256_sub_ps(vb11, v_zp1);

                __m256 vb20, vb21;
                mm256_loadu_u4_to_f32(b_data_ptr + i / 2 + 2 * src_stride, vb20, vb21);
                vb20 = _mm256_sub_ps(vb20, v_zp2);
                vb21 = _mm256_sub_ps(vb21, v_zp2);

                __m256 vb30, vb31;
                mm256_loadu_u4_to_f32(b_data_ptr + i / 2 + 3 * src_stride, vb30, vb31);
                vb30 = _mm256_sub_ps(vb30, v_zp3);
                vb31 = _mm256_sub_ps(vb31, v_zp3);

                vsum0 = _mm256_fmadd_ps(va0, vb00, vsum0);
                vsum0 = _mm256_fmadd_ps(va1, vb01, vsum0);

                vsum1 = _mm256_fmadd_ps(va0, vb10, vsum1);
                vsum1 = _mm256_fmadd_ps(va1, vb11, vsum1);

                vsum2 = _mm256_fmadd_ps(va0, vb20, vsum2);
                vsum2 = _mm256_fmadd_ps(va1, vb21, vsum2);

                vsum3 = _mm256_fmadd_ps(va0, vb30, vsum3);
                vsum3 = _mm256_fmadd_ps(va1, vb31, vsum3);
            }
            hsum(vsum0);
            hsum(vsum1);
            hsum(vsum2);
            hsum(vsum3);
            float group_sum0 = _mm256_cvtss_f32(vsum0);
            float group_sum1 = _mm256_cvtss_f32(vsum1);
            float group_sum2 = _mm256_cvtss_f32(vsum2);
            float group_sum3 = _mm256_cvtss_f32(vsum3);
            for (; i < group_size; i += 2) {
                uint8_t data = b_data_ptr[i / 2];
                float tmp0 = extract_half_byte(data, static_cast<bool>(i % 2));
                float tmp1 = extract_half_byte(data, static_cast<bool>((i + 1) % 2));
                group_sum0 += a[dst_offset + i] * (tmp0 - b0[1]);
                group_sum0 += a[dst_offset + i + 1] * (tmp1 - b0[1]);

                uint8_t data1 = b_data_ptr[i / 2 + src_stride];
                float tmp10 = extract_half_byte(data1, static_cast<bool>(i % 2));
                float tmp11 = extract_half_byte(data1, static_cast<bool>((i + 1) % 2));
                group_sum1 += a[dst_offset + i] * (tmp10 - b1[1]);
                group_sum1 += a[dst_offset + i + 1] * (tmp11 - b1[1]);

                uint8_t data2 = b_data_ptr[i / 2 + 2 * src_stride];
                float tmp20 = extract_half_byte(data2, static_cast<bool>(i % 2));
                float tmp21 = extract_half_byte(data2, static_cast<bool>((i + 1) % 2));
                group_sum2 += a[dst_offset + i] * (tmp20 - b2[1]);
                group_sum2 += a[dst_offset + i + 1] * (tmp21 - b2[1]);

                uint8_t data3 = b_data_ptr[i / 2 + 3 * src_stride];
                float tmp30 = extract_half_byte(data3, static_cast<bool>(i % 2));
                float tmp31 = extract_half_byte(data3, static_cast<bool>((i + 1) % 2));
                group_sum3 += a[dst_offset + i] * (tmp30 - b3[1]);
                group_sum3 += a[dst_offset + i + 1] * (tmp31 - b3[1]);
            }
            sum0 += group_sum0 * b0[0];
            sum1 += group_sum1 * b1[0];
            sum2 += group_sum2 * b2[0];
            sum3 += group_sum3 * b3[0];
            dst_offset += group_size;
            src_offset += group_size + params_offset;
        }
        c[0] = sum0;
        c[1] = sum1;
        c[2] = sum2;
        c[3] = sum3;
        c += 4;
        b += 4 * src_stride;
    }
#endif

    for (; j < block_size; j++) {
        float sum = 0;
        dst_offset = 0;
        src_offset = 0;
        while (dst_offset < n) {
            auto b0 = reinterpret_cast<float*>(b + src_offset);
            float group_sum = 0.0f;
            for (size_t i = 0; i < group_size; i += 2) {
                uint8_t data = b[i / 2 + src_offset + params_offset];
                float tmp0 = extract_half_byte(data, static_cast<bool>(i % 2));
                float tmp1 = extract_half_byte(data, static_cast<bool>((i + 1) % 2));
                group_sum += a[dst_offset + i] * (tmp0 - b0[1]);
                group_sum += a[dst_offset + i + 1] * (tmp1 - b0[1]);
            }
            sum += group_sum * b0[0];
            dst_offset += group_size;
            src_offset += group_size / sub_byte_multiplier + params_offset;
        }
        b += src_stride;
        *c++ = sum;
    }
}

template <
    typename TA,
    ov::element::Type_t SRC_PREC,
    std::enable_if_t<(ov::intel_cpu::any_of(SRC_PREC, ov::element::i8, ov::element::u8, ov::element::u4)), bool> = true>
void dot_product_block_quantized(TA* a,
                                 uint8_t* b,
                                 float* c,
                                 const size_t n,
                                 const bool is_bychannel,
                                 const size_t block_size,
                                 const size_t group_size) {
    if (is_bychannel) {
        if constexpr (ov::intel_cpu::any_of(SRC_PREC, ov::element::u8, ov::element::u4)) {
            dot_product_block_quantized_by_channel<TA, SRC_PREC>(a, b, c, n, block_size);
        }
    } else {
        dot_product_block_quantized_by_dims<TA, SRC_PREC>(a, b, c, n, block_size, group_size);
    }
}

template <typename T>
void attn_reduce(T* dst, float* temp, size_t M, size_t S, size_t temp_stride) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
        auto* src = temp + i;
        auto result_vec_fp32 = _mm512_setzero_ps();
        for (size_t m = 0; m < M; m++) {
            auto o_vec_fp32 = _mm512_loadu_ps(src);
            result_vec_fp32 = _mm512_add_ps(result_vec_fp32, o_vec_fp32);
            src += temp_stride;
        }
        // save to bf16
        mm512_uni_storeu_ps(dst + i, result_vec_fp32);
    }
#elif defined(HAVE_AVX2)
    for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
        auto* src = temp + i;
        auto result_vec_fp32 = _mm256_set1_ps(0.0f);
        for (size_t m = 0; m < M; m++) {
            auto o_vec_fp32 = mm256_uni_loadu_ps(src);
            result_vec_fp32 = _mm256_add_ps(result_vec_fp32, o_vec_fp32);
            src += temp_stride;
        }
        mm256_uni_storeu_ps(dst + i, result_vec_fp32);
    }
#endif
    for (; i < S; i++) {
        auto* src = temp + i;
        float sum = 0.0f;
        // sum result from all threads partition
        for (size_t m = 0; m < M; m++) {
            sum += src[0];
            src += temp_stride;
        }
        dst[i] = sum;
    }
}

}  // namespace ov::Extensions::Cpu::XARCH