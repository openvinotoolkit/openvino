// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include "attn_memcpy.hpp"
#include "attn_quant.hpp"
#include "attn_quant_kernel.hpp"
#include "cache_rotation.hpp"
#include "common.hpp"
#include "executor_pa.hpp"
#include "executor_pa_common.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "softmax_kernel.hpp"
#include "transpose_kernel.hpp"
#include "utils/plain_tensor.hpp"
#if defined(OPENVINO_ARCH_X86_64)
#    include "nodes/kernels/x64/brgemm_kernel.hpp"
#elif defined(OPENVINO_ARCH_ARM64) && defined(HAVE_SVE)
#    include "arm_sve.h"
#    include "nodes/kernels/aarch64/brgemm_kernel.hpp"
#endif

namespace ov::Extensions::Cpu::XARCH {

using namespace ov;
using namespace ov::intel_cpu;

// currently depends on brgemm which only support x64
#if defined(OPENVINO_ARCH_X86_64) || (defined(OPENVINO_ARCH_ARM64) && defined(HAVE_SVE))

#    if defined(HAVE_AVX2) || defined(HAVE_AVX512F)

#        define prefetch_bytes(bytes, sel, advance, src) \
            {                                            \
                auto* p = reinterpret_cast<char*>(src);  \
                for (size_t i = 0; i < bytes; i += 64)   \
                    _mm_prefetch(p + i + advance, sel);  \
            }

#    else

#        define prefetch_bytes(bytes, sel, advance, src)

#    endif

size_t inline get_sub_byte_multiplier(ov::element::Type type) {
    return one_of(type, ov::element::i4, ov::element::u4) ? 8 / type.bitwidth() : 1;
}

template <typename T,
          ov::element::Type_t SRC_PREC,
          std::enable_if_t<(std::is_same_v<T, ov::bfloat16> || std::is_same_v<T, ov::float16> ||
                            std::is_same_v<T, float>)&&(SRC_PREC != ov::element::u8 || SRC_PREC != ov::element::u4),
                           bool> = true>
static void attn_acc_value_block(float* out,
                                 float* weight,
                                 T* v,
                                 const size_t S,
                                 const size_t block_size,
                                 const size_t group_size) {
#    if defined(HAVE_AVX512F)
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
#    elif defined(HAVE_AVX2)
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
#    endif
    for (size_t j = 0; j < block_size; j++) {
        for (size_t i = 0; i < S; i++) {
            out[i] += weight[j] * v[i];
        }
        v += S;
    }
}
template <typename T, ov::element::Type_t SRC_PREC, std::enable_if_t<SRC_PREC == ov::element::u8, bool> = true>
static void attn_acc_value_block(float* out,
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

#    if defined(HAVE_AVX512F)
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
#    elif defined(HAVE_AVX2)
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
#    endif
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
}

template <typename T, ov::element::Type_t SRC_PREC, std::enable_if_t<SRC_PREC == ov::element::u4, bool> = true>
static void attn_acc_value_block(float* out,
                                 float* weight,
                                 void* v,
                                 const size_t S,
                                 const size_t block_size,
                                 const size_t group_size) {
    size_t src_offset = 0;
    size_t dst_offset = 0;
    const size_t params_offset = sizeof(float) * 2;
    auto* v_ptr = reinterpret_cast<uint8_t*>(v);
    auto sub_byte_multiplier = 8 / 4;
    const size_t src_stride = S / group_size * (group_size / sub_byte_multiplier + params_offset);
    auto extract_half_byte = [](uint8_t val, bool high_half) -> uint8_t {
        uint8_t shift = high_half ? 0 : 4;

        return static_cast<uint8_t>((val >> shift) & 0x000F);
    };
    for (size_t j = 0; j < block_size; j++) {
        dst_offset = 0;
        src_offset = 0;
        while (dst_offset < S) {
            auto v0 = reinterpret_cast<float*>(v_ptr + src_offset);
            size_t i = 0;
#    if defined(HAVE_AVX512F)
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
#    elif defined(HAVE_AVX2)
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
#    endif
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

template <typename TA, typename TB>
static void dot_product_block(TA* a,
                              TB* b,
                              float* c,
                              const size_t n,
                              const size_t block_size,
                              const size_t group_size) {
#    if defined(HAVE_AVX512F)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto vsum0 = _mm512_setzero_ps();
        auto vsum1 = _mm512_setzero_ps();
        auto vsum2 = _mm512_setzero_ps();
        auto vsum3 = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
            auto va = mm512_uni_loadu_ps(a + i);
            vsum0 = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b + i), vsum0);
            vsum1 = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b + i + n), vsum1);
            vsum2 = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b + i + 2 * n), vsum2);
            vsum3 = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b + i + 3 * n), vsum3);
        }
        float sum0 = _mm512_reduce_add_ps(vsum0);
        float sum1 = _mm512_reduce_add_ps(vsum1);
        float sum2 = _mm512_reduce_add_ps(vsum2);
        float sum3 = _mm512_reduce_add_ps(vsum3);
        for (; i < n; i++) {
            sum0 += a[i] * b[i];
            sum1 += a[i] * b[i + n];
            sum2 += a[i] * b[i + 2 * n];
            sum3 += a[i] * b[i + 3 * n];
        }
        c[0] = sum0;
        c[1] = sum1;
        c[2] = sum2;
        c[3] = sum3;
        c += 4;
        b += 4 * n;
    }
    for (; j < block_size; j++) {
        auto vsum = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
            auto va = mm512_uni_loadu_ps(a + i);
            vsum = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b + i), vsum);
        }
        float sum = _mm512_reduce_add_ps(vsum);
        for (; i < n; i++) {
            sum += a[i] * b[i];
        }
        b += n;
        *c++ = sum;
    }
    return;
#    elif defined(HAVE_AVX2)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto vsum0 = _mm256_set1_ps(0.0f);
        auto vsum1 = _mm256_set1_ps(0.0f);
        auto vsum2 = _mm256_set1_ps(0.0f);
        auto vsum3 = _mm256_set1_ps(0.0f);
        size_t i = 0;
        for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
            auto va = mm256_uni_loadu_ps(a + i);
            vsum0 = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b + i), vsum0);
            vsum1 = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b + i + n), vsum1);
            vsum2 = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b + i + 2 * n), vsum2);
            vsum3 = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b + i + 3 * n), vsum3);
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
            sum0 += a[i] * b[i];
            sum1 += a[i] * b[i + n];
            sum2 += a[i] * b[i + 2 * n];
            sum3 += a[i] * b[i + 3 * n];
        }
        c[0] = sum0;
        c[1] = sum1;
        c[2] = sum2;
        c[3] = sum3;
        c += 4;
        b += 4 * n;
    }
    for (; j < block_size; j++) {
        auto vsum = _mm256_set1_ps(0.0f);
        size_t i = 0;
        for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
            auto va = mm256_uni_loadu_ps(a + i);
            vsum = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b + i), vsum);
        }
        hsum(vsum);
        float sum = _mm256_cvtss_f32(vsum);
        for (; i < n; i++) {
            sum += a[i] * b[i];
        }
        b += n;
        *c++ = sum;
    }
    return;
#    endif
    for (size_t j = 0; j < block_size; j++) {
        float sum = 0;
        for (size_t i = 0; i < n; i++) {
            sum += a[i] * b[i];
        }
        b += n;
        *c++ = sum;
    }
}

template <typename TA>
static void dot_product_block_by_channel(TA* a, uint8_t* b, float* c, const size_t n, const size_t block_size) {
    const size_t params_offset = sizeof(float) * 2 * n;
    const size_t src_stride = n;
    auto p_scales = reinterpret_cast<float*>(b);
    auto p_zps = p_scales + n;
    for (size_t j = 0; j < block_size; j++) {
        float sum = 0.0f;
        size_t i = 0;
#    if defined(HAVE_AVX512F)
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
#    endif
#    if defined(HAVE_AVX2)
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
#    endif
        for (; i < n; i++) {
            sum += a[i] * (b[params_offset + i] - p_zps[i]) * p_scales[i];
        }
        b += src_stride;
        *c++ = sum;
    }
}

template <typename TA>
static void dot_product_block(TA* a,
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
    const size_t src_stride = n / group_size * (group_size + params_offset);
#    if defined(HAVE_AVX512F)
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
#    elif defined(HAVE_AVX2)
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
#    endif
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
}

template <typename T>
static void attn_reduce(T* dst, float* temp, size_t M, size_t S, size_t temp_stride) {
    size_t i = 0;
#    if defined(HAVE_AVX512F)
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
#    elif defined(HAVE_AVX2)
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
#    endif
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

// N must be multiple of 16
template <typename TDST,
          ov::element::Type_t SRC_PREC,
          std::enable_if_t<(SRC_PREC != ov::element::u8 && SRC_PREC != ov::element::u4), bool> = true>
void transpose_16NxK(TDST* dst,
                     void* src,
                     TDST* tmp,
                     const size_t N,
                     const size_t K,
                     const size_t dst_stride,
                     const size_t src_stride,
                     const size_t group_size,
                     const bool quant_key_bychannel) {
    size_t k = 0;
    auto* src_ptr = reinterpret_cast<typename ov::element_type_traits<SRC_PREC>::value_type*>(src);
    for (; k + 16 <= K; k += 16) {
        for (size_t n = 0; n < N; n += 16) {
            transpose_16x16_kernel(dst + n, src_ptr + n * src_stride, dst_stride, src_stride);
        }

        dst += 16 * dst_stride;
        src_ptr += 16;
    }
    if (k < K) {
        for (size_t n = 0; n < N; n += 16) {
            transpose_16xK_kernel(dst + n, src_ptr + n * src_stride, K - k, dst_stride, src_stride);
        }
    }
}
#    if defined(HAVE_AVX512F)
template <typename T,
          ov::element::Type_t SRC_PREC,
          typename std::enable_if<(SRC_PREC == ov::element::bf16 || SRC_PREC == ov::element::f16) &&
                                      (SRC_PREC == precision_of<T>::value),
                                  bool>::type = true>
static void transpose_16NxK(T* dst,
                            T* src,
                            T* tmp,
                            const size_t N,
                            const size_t K,
                            const size_t dst_stride,
                            const size_t src_stride,
                            const size_t group_size,
                            const bool quant_key_bychannel) {
    // will treat as uint32_t transpose
    auto s = reinterpret_cast<uint32_t*>(src);
    auto d = reinterpret_cast<uint32_t*>(dst);
    transpose_16NxK<uint32_t, ov::element::u32>(d,
                                                s,
                                                reinterpret_cast<uint32_t*>(0),
                                                N,
                                                K >> 1,
                                                dst_stride,
                                                src_stride >> 1,
                                                group_size,
                                                false);
}
#    endif

template <typename TDST, ov::element::Type_t SRC_PREC, std::enable_if_t<SRC_PREC == ov::element::u8, bool> = true>
void transpose_16NxK(TDST* dst,
                     void* src,
                     TDST* tmp,
                     const size_t N,
                     const size_t K,
                     const size_t dst_stride,
                     const size_t src_stride,
                     const size_t group_size,
                     const bool quant_key_bychannel) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized
    // feature(u8,idx_S)| The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
    auto s = reinterpret_cast<typename ov::element_type_traits<SRC_PREC>::value_type*>(src);
    auto t = tmp;
    // if group_size not set, the whole row is used as a group
    if (quant_key_bychannel) {
        auto p_scales = reinterpret_cast<float*>(s);
        auto p_zps = p_scales + K;
        s = s + sizeof(float) * 2 * K;
        attn_dequant_u8_by_channel_kernel(s, t, N, K, K, src_stride, p_scales, p_zps);
    } else {
        for (size_t n = 0; n < N; n++) {
            size_t src_offset = 0;
            size_t dst_offset = 0;
            while (dst_offset < K) {
                auto f = reinterpret_cast<float*>(s + src_offset);
                attn_dequant_kernel<TDST, SRC_PREC>(s + src_offset + sizeof(float) * 2,
                                                    t + dst_offset,
                                                    group_size,
                                                    f[0],
                                                    f[1]);
                src_offset += group_size + sizeof(float) * 2;
                dst_offset += group_size;
            }
            s += src_offset;
            t += src_stride;
        }
    }
    transpose_16NxK<TDST, precision_of<TDST>::value>(dst,
                                                     tmp,
                                                     reinterpret_cast<TDST*>(0),
                                                     N,
                                                     K,
                                                     dst_stride,
                                                     src_stride,
                                                     0,
                                                     false);
}

// dequant f16/u8 to float
template <typename T,
          ov::element::Type_t SRC_PREC,
          std::enable_if_t<SRC_PREC != ov::element::u8 && precision_of<T>::value == SRC_PREC, bool> = true>
static inline void dequant(T* dst, void* src, const size_t N, const size_t K, const size_t group_size) {
    // never called
    OPENVINO_THROW("dequant: should not be called.");
}
template <typename T, ov::element::Type_t SRC_PREC, std::enable_if_t<SRC_PREC == ov::element::f16, bool> = true>
static inline void dequant(float* dst, void* src, const size_t N, const size_t K, const size_t group_size) {
    cvt_copy(dst, reinterpret_cast<ov::float16*>(src), 1, K * N, 0, 0);
}

template <typename TDST,
          ov::element::Type_t SRC_PREC,
          std::enable_if_t<SRC_PREC == ov::element::u4 || SRC_PREC == ov::element::u8, bool> = true>
void dequant(TDST* dst, uint8_t* src, const size_t N, const size_t K, const size_t group_size) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized
    // feature(u8,idx_S)| The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
    auto s = src;
    const size_t params_offset = sizeof(float) * 2;
    const size_t sub_byte_mulitplier = get_sub_byte_multiplier(SRC_PREC);

    for (size_t n = 0; n < N; n++) {
        size_t src_offset = 0;
        size_t dst_offset = 0;
        while (dst_offset < K) {
            auto f = reinterpret_cast<float*>(s + src_offset);
            attn_dequant_kernel<TDST, SRC_PREC>(s + src_offset + params_offset,
                                                dst + dst_offset,
                                                group_size,
                                                f[0],
                                                f[1]);
            src_offset += group_size / sub_byte_mulitplier + params_offset;
            dst_offset += group_size;
        }
        s += src_offset;
        dst += K;
    }
}

#    if defined(HAVE_AVX512F)
template <typename T,
          typename = typename std::
              enable_if<(std::is_same<T, ov::bfloat16>::value || std::is_same<T, ov::float16>::value), bool>::type>
static void pack_32x32_kernel(T* dst, T* src, size_t dst_stride, size_t src_stride) {
    static const uint64_t idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    auto midx = _mm512_loadu_si512(idx);
    for (size_t i = 0; i < 16; i++) {
        auto a = _mm512_loadu_si512(src);  // [a1  a2  a3 a4 | a5  a6  a7 a8]   total 512-bits in 8 64bits unit
        auto b = _mm512_loadu_si512(src + src_stride);  // [b1  b2  b3 b4 | b5  b6  b7 b8]   total 512-bits
        a = _mm512_permutexvar_epi64(midx, a);          // [a1 a5 | a2 a6 | a3 a7 | a4 a8]
        b = _mm512_permutexvar_epi64(midx, b);          // [b1 b5 | b2 b6 | b3 b7 | b4 b8]
        auto B0 = _mm512_unpacklo_epi16(
            a,
            b);  // [ a1&b1  a2&b2   a3&b3   a4&b4] for each 128-bits lane, interleave word in low 64 bits
        auto B1 = _mm512_unpackhi_epi16(
            a,
            b);  // [ a5&b5  a6&b6   a7&b7   a8&b8] for each 128-bits lane, interleave word in high 64 bits
        _mm512_storeu_si512(dst, B0);
        _mm512_storeu_si512(dst + 32, B1);
        src += 2 * src_stride;
        dst += 2 * dst_stride;
    }
}

template <typename T,
          typename = typename std::
              enable_if<(std::is_same<T, ov::bfloat16>::value || std::is_same<T, ov::float16>::value), bool>::type>
static void pack_32x16_kernel(T* dst, T* src, size_t dst_stride, size_t src_stride) {
    static const uint64_t idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    auto midx = _mm512_loadu_si512(idx);
    for (size_t i = 0; i < 16; i++) {
        auto x =
            _mm256_loadu_si256(reinterpret_cast<__m256i*>(src));  // [a1  a2  a3 a4]   total 256-bits in 4 64bits unit
        auto y = _mm256_loadu_si256(reinterpret_cast<__m256i*>(src + src_stride));  // [b1  b2  b3 b4]   total 256-bits
        auto a = _mm512_castsi256_si512(x);
        auto b = _mm512_castsi256_si512(y);
        a = _mm512_permutexvar_epi64(midx, a);  // [a1 x | a2 x | a3 x | a4 x]
        b = _mm512_permutexvar_epi64(midx, b);  // [b1 x | b2 x | b3 x | b4 x]
        auto B0 = _mm512_unpacklo_epi16(a, b);
        _mm512_storeu_si512(dst, B0);
        src += 2 * src_stride;
        dst += 2 * dst_stride;
    }
}

template <typename T,
          typename = typename std::
              enable_if<(std::is_same<T, ov::bfloat16>::value || std::is_same<T, ov::float16>::value), bool>::type>
static void pack_32xK_kernel(T* dst, T* src, size_t dst_stride, size_t src_stride, size_t K) {
    static const uint64_t idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    auto midx = _mm512_loadu_si512(idx);
    __mmask16 mask = (1 << K) - 1;
    for (size_t i = 0; i < 16; i++) {
        auto x = _mm256_maskz_loadu_epi16(mask, src);               // [a1  a2  a3 a4]   total 256-bits in 4 64bits unit
        auto y = _mm256_maskz_loadu_epi16(mask, src + src_stride);  // [b1  b2  b3 b4]   total 256-bits
        auto a = _mm512_castsi256_si512(x);
        auto b = _mm512_castsi256_si512(y);
        a = _mm512_permutexvar_epi64(midx, a);  // [a1 x | a2 x | a3 x | a4 x]
        b = _mm512_permutexvar_epi64(midx, b);  // [b1 x | b2 x | b3 x | b4 x]
        auto B0 = _mm512_unpacklo_epi16(a, b);
        _mm512_mask_storeu_epi32(dst, mask, B0);
        src += 2 * src_stride;
        dst += 2 * dst_stride;
    }
}

template <typename TDST,
          ov::element::Type_t SRC_PREC,
          typename std::enable_if<precision_of<TDST>::value != ov::element::f32 &&
                                      (SRC_PREC == ov::element::bf16 || SRC_PREC == ov::element::f16),
                                  bool>::type = true>
static void pack_32NxK(TDST* dst,
                       void* src,
                       TDST* tmp,
                       const size_t N,
                       const size_t K,
                       const size_t dst_stride,
                       const size_t src_stride,
                       const size_t group_size) {
    auto src_ptr = reinterpret_cast<typename ov::element_type_traits<SRC_PREC>::value_type*>(src);
    for (size_t n = 0; n < N; n += 32) {
        size_t k = 0;
        for (; k + 32 <= K; k += 32) {
            pack_32x32_kernel(dst + k * 2, src_ptr + k, dst_stride, src_stride);
        }
        if (k + 16 <= K) {
            pack_32x16_kernel(dst + k * 2, src_ptr + k, dst_stride, src_stride);
            k += 16;
        }
        if (k < K) {
            pack_32xK_kernel(dst + k * 2, src_ptr + k, dst_stride, src_stride, K - k);
        }

        dst += 32 * dst_stride;
        src_ptr += 32 * src_stride;
    }
}

template <typename TDST,
          ov::element::Type_t SRC_PREC,
          typename std::enable_if<precision_of<TDST>::value != ov::element::f32 &&
                                      (SRC_PREC == ov::element::u4 || SRC_PREC == ov::element::u8),
                                  bool>::type = true>
static void pack_32NxK(TDST* dst,
                       void* src,
                       TDST* tmp,
                       const size_t N,
                       const size_t K,
                       const size_t dst_stride,
                       const size_t src_stride,
                       const size_t group_size) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized
    // feature(u8,idx_S)| The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
    auto s = reinterpret_cast<uint8_t*>(src);
    auto t = tmp;
    // if group_size not set, the whole row is used as a group
    const size_t sub_byte_mulitplier = get_sub_byte_multiplier(SRC_PREC);
    for (size_t n = 0; n < N; n++) {
        size_t src_offset = 0;
        size_t dst_offset = 0;
        while (dst_offset < K) {
            auto f = reinterpret_cast<float*>(s + src_offset);
            attn_dequant_kernel<TDST, SRC_PREC>(s + (src_offset + sizeof(float) * 2),
                                                t + dst_offset,
                                                group_size,
                                                f[0],
                                                f[1]);
            src_offset += group_size / sub_byte_mulitplier + sizeof(float) * 2;
            dst_offset += group_size;
        }
        s += src_offset;
        t += src_stride;
    }
    pack_32NxK<TDST, precision_of<TDST>::value>(dst,
                                                tmp,
                                                reinterpret_cast<TDST*>(0),
                                                N,
                                                K,
                                                dst_stride,
                                                src_stride,
                                                group_size);
}
#    endif

template <typename TDST,
          ov::element::Type_t SRC_PREC,
          std::enable_if_t<precision_of<TDST>::value == ov::element::f32, bool> = true>
static void pack_32NxK(TDST* dst,
                       void* src,
                       TDST* tmp,
                       const size_t N,
                       const size_t K,
                       const size_t dst_stride,
                       const size_t src_stride,
                       const size_t group_size) {
    // never called
    OPENVINO_THROW("pack_32NxK: should not be called.");
}

template <class T>
void fill_rotation_coefficients_from_lut(T* rotation_coefficients_block_data,
                                         const int32_t* rotation_deltas_block_data,
                                         size_t rotation_deltas_token_stride,
                                         const T* rotation_trig_lut,
                                         size_t block_size,
                                         size_t embedding_size) {
    size_t dst_offset = 0;
    for (size_t tok_idx = 0; tok_idx < block_size; tok_idx++) {
        size_t gather_idx = *(rotation_deltas_block_data + rotation_deltas_token_stride * tok_idx);
        size_t src_offset = gather_idx * embedding_size;
        std::memcpy(rotation_coefficients_block_data + dst_offset,
                    rotation_trig_lut + src_offset,
                    embedding_size * sizeof(T));
        dst_offset += embedding_size;
    }
}

template <class KVCACHE_TYPE>
void rotate_kv_cache(PlainTensor& key_cache,
                     const PlainTensor& rotated_block_indices,
                     const PlainTensor& rotation_deltas,
                     const PlainTensor& rotation_trig_lut,
                     PlainTensor& rotation_coefficients_scratch) {
    size_t num_blocks_in_total = key_cache.size(0);
    size_t num_heads = key_cache.size(1);  // H;
    size_t block_size = key_cache.size(2);
    size_t embedding_size = key_cache.size(3);  // S;

    size_t num_rotated_blocks = rotated_block_indices.size(0);
    auto* rotated_block_indices_data = rotated_block_indices.ptr<int32_t>();
    auto* rotation_trig_lut_data = rotation_trig_lut.ptr<float>();

    size_t rotation_deltas_token_stride = 0;
    size_t rotation_deltas_block_stride = 1;

    bool is_per_token = (rotation_deltas.shape()[1] == block_size);
    if (is_per_token) {
        rotation_deltas_token_stride = 1;
        rotation_deltas_block_stride = block_size;
    }

    for (size_t i = 0; i < num_rotated_blocks; i++) {
        size_t rotated_block_index = *(rotated_block_indices_data + i);
        OPENVINO_ASSERT(rotated_block_index < num_blocks_in_total);

        int32_t* rotation_deltas_block_data = rotation_deltas.ptr<int32_t>() + i * rotation_deltas_block_stride;

        auto* rotation_coefficient_block_data = rotation_coefficients_scratch.ptr<float>();
        fill_rotation_coefficients_from_lut(rotation_coefficient_block_data,
                                            rotation_deltas_block_data,
                                            rotation_deltas_token_stride,
                                            rotation_trig_lut_data,
                                            block_size,
                                            embedding_size);
        auto* cache_block_ptr = key_cache.ptr<KVCACHE_TYPE>(rotated_block_index);
        rotate_kv_cache_block(cache_block_ptr, rotation_coefficient_block_data, num_heads, block_size, embedding_size);
    }
}

template <typename DATA_TYPE, typename KEY_CACHE_TYPE, ov::element::Type_t VALUE_PREC>
struct MHAHelper {
    // initialize once
    size_t H;
    size_t S;
    size_t SV;
    size_t Hk;
    size_t _h_each_group_len;
    size_t _block_size;
    size_t _nthr;
    size_t _sliding_window;
    float _d_scale;
    size_t _key_group_size = 0;
    size_t _value_group_size = 0;
    bool _quant_key_bychannel = 0;

    PlainTensor _weight;        // [nthr, H, 32, rnd_up(kv_len, block_size)], shared by first and second loop along bh
    PlainTensor _output;        // [nthr, 32, H, S], shared by first and second loop along bh
    PlainTensor _qk_scratch_a;  // [nthr, scratch_a_size]
    PlainTensor _qk_scratch_b;  // [B, rnd_up(kv_len, block_size), Hk, scratch_b_size]
    PlainTensor _wv_scratch_a;
    PlainTensor _wv_scratch_b;
    PlainTensor _alibi_lookup;
    PlainTensor _score_output;
    std::vector<size_t> _wsp;
    size_t _wsp_size_per_thread = 0;

    std::vector<std::shared_ptr<BrgemmKernel>> _qk_gemm;
    std::vector<std::shared_ptr<BrgemmKernel>> _wv_gemm;
    // will accumulate C buffer
    std::vector<std::shared_ptr<BrgemmKernel>> _wv_gemm_acc;
// second token
#    if defined(OPENVINO_ARCH_X86_64)
    std::shared_ptr<JitMatMulVecAMX> _gemv;
#    endif
    ov::element::Type _fastpath_valid_prec = ov::element::dynamic;
    // second token for bhl loop
    PlainTensor _weight_bhl;
    PlainTensor _output_bhl;
    PlainTensor _score_offsets_aligned;
    PlainTensor _score_offsets;

    PlainTensor _block_rotation_coefficient_scratch;

    MHAHelper() {
        _weight.resize<float>({size_t{1}, size_t{1}, size_t{1}, size_t{1}});
    }

    explicit MHAHelper(size_t key_group_size, size_t value_group_size, bool quant_key_bychannel)
        : _key_group_size(key_group_size),
          _value_group_size(value_group_size),
          _quant_key_bychannel(quant_key_bychannel) {
        _weight.resize<float>({size_t{1}, size_t{1}, size_t{1}, size_t{1}});
    }

    void init(size_t H,
              size_t S,
              size_t SV,
              size_t Hk,
              size_t h_each_group_len,
              size_t block_size,
              size_t sliding_window,
              float d_scale,
              size_t kv_len,
              bool init_alibi_lookup,
              bool init_rotation_coefficient_scratch) {
        // query shape: [B, H, L, S]
        // present_key shape: [block, H, 32, S]
        // Q*K': [M1, S] * [M2, S]'
        //   kernel: Q:[1~block_size, S] * K':[block_size, S]'
        //   aka: M:1~block_size, N:block_size, K:S
        // (Q*K')*V: [M1, M2] * [M2, S]
        //   kernel: (Q*K'):[1~block_size, block_size] * V:[block_size, S]
        //   aka: M:1~block_size, N:S, K:block_size
        // Because K and V are from cache, can use M2'=rnd_up(M2, block_size) to simplify logic
        auto in_type = precision_of<DATA_TYPE>::value;
        this->H = H;
        this->S = S;
        this->SV = SV;
        this->Hk = Hk;
        _h_each_group_len = h_each_group_len;
        _block_size = block_size;
        _nthr = static_cast<size_t>(parallel_get_max_threads());
        _sliding_window = sliding_window;
        _d_scale = d_scale;

        auto prev_score_stride = _weight.stride(2);
        auto want_score_stride = rnd_up(kv_len, _block_size);
        auto new_score_stride = std::max(prev_score_stride, want_score_stride);
        // resize temporary buffers, weight.size(3) will be aligned to block_size
        _weight.resize<float>({static_cast<size_t>(_nthr), H, _block_size, new_score_stride});
        // std::max(S, SV) here is to ensure by_channel quantize has enough buffer to use
        if (_quant_key_bychannel)
            _output.resize<float>({static_cast<size_t>(_nthr), _block_size, H, std::max(S, SV)});
        else
            _output.resize<float>({static_cast<size_t>(_nthr), _block_size, H, SV});

        // TODO: kernel supports stride
        if (_qk_gemm.empty() || prev_score_stride < new_score_stride) {
            _qk_gemm.resize(_block_size);
            _wv_gemm.resize(_block_size);
            _wv_gemm_acc.resize(_block_size);
            for (size_t i = 0; i < _block_size; i++) {
                _qk_gemm[i] = std::make_shared<BrgemmKernel>(i + 1,
                                                             _block_size,
                                                             S,
                                                             H * S,
                                                             _block_size,
                                                             _weight.stride(2),
                                                             false,
                                                             in_type);
                _wv_gemm[i] =
                    std::make_shared<BrgemmKernel>(i + 1,
                                                   SV,
                                                   _block_size,
                                                   // if it's bf16, the stride needs double due to reuse float buffer
                                                   (in_type == ov::element::Type_t::f32 ? 1 : 2) * _weight.stride(2),
                                                   SV,
                                                   _output.stride(1),
                                                   false,
                                                   in_type);
                _wv_gemm_acc[i] =
                    std::make_shared<BrgemmKernel>(i + 1,
                                                   SV,
                                                   _block_size,
                                                   // if it's bf16, the stride needs double due to reuse float buffer
                                                   (in_type == ov::element::Type_t::f32 ? 1 : 2) * _weight.stride(2),
                                                   SV,
                                                   _output.stride(1),
                                                   false,
                                                   in_type,
                                                   true);
            }

            // wsp is used to compute beta when K is blocked
            _wsp_size_per_thread = _wv_gemm[0]->get_wsp_size();
            _wsp.resize(_nthr * _wsp_size_per_thread);

            // allocate scratch a/b, notice get_scratch_a_size/get_scratch_b_size returns in bytes
            _qk_scratch_a.resize<DATA_TYPE>(
                {_nthr, _qk_gemm[_block_size - 1]->get_scratch_a_size() / sizeof(DATA_TYPE)});
            _wv_scratch_a.resize<DATA_TYPE>(
                {_nthr, _wv_gemm[_block_size - 1]->get_scratch_a_size() / sizeof(DATA_TYPE)});

#    if defined(OPENVINO_ARCH_X86_64)
            if ((S % 32 == 0) && (block_size % 16 == 0) && (S <= 32 * 6)) {
                if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::amx_bf16) &&
                    precision_of<DATA_TYPE>::value == ov::element::bf16 &&
                    precision_of<KEY_CACHE_TYPE>::value == ov::element::bf16 && VALUE_PREC == ov::element::bf16) {
                    _fastpath_valid_prec = ov::element::bf16;
                } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::amx_fp16) &&
                           precision_of<DATA_TYPE>::value == ov::element::f16 &&
                           precision_of<KEY_CACHE_TYPE>::value == ov::element::f16 && VALUE_PREC == ov::element::f16) {
                    _fastpath_valid_prec = ov::element::f16;
                }
            }
            if (one_of(_fastpath_valid_prec, ov::element::bf16, ov::element::f16) && !_gemv) {
                _gemv = std::make_shared<JitMatMulVecAMX>(static_cast<int>(S),
                                                          static_cast<int>(block_size),
                                                          _fastpath_valid_prec);
            }
#    endif
        }

        if (init_alibi_lookup && (!_alibi_lookup || _alibi_lookup.m_dims[0] < kv_len)) {
            _alibi_lookup.resize<float>({kv_len * 2});
            for (size_t i = 0; i < _alibi_lookup.m_dims[0]; i++) {
                _alibi_lookup.ptr<float>()[i] = -static_cast<int>((_alibi_lookup.m_dims[0] - 1 - i));
            }
        }

        if (init_rotation_coefficient_scratch) {
            _block_rotation_coefficient_scratch.resize<DATA_TYPE>({_block_size, S});
        }
    }

    void init_reorder_buffers(size_t batch, size_t kv_len_in_blocks) {
        _qk_scratch_b.resize<DATA_TYPE>({batch, kv_len_in_blocks, Hk, _block_size * S});
        _wv_scratch_b.resize<DATA_TYPE>({batch, kv_len_in_blocks, Hk, _block_size * rnd_up(SV, _block_size)});
    }

    void init_score_buffers(const PlainTensor& past_lens, const PlainTensor& subsequence_begins) {
        static constexpr int cache_line_size = dnnl::impl::cpu::platform::get_cache_line_size();
        auto seq_cout = static_cast<int32_t>(past_lens.m_dims[0]);
        _score_offsets_aligned.resize<int32_t>({past_lens.m_dims[0]});
        _score_offsets.resize<int32_t>({past_lens.m_dims[0]});
        int32_t total_kv_len_aligned = 0;
        int32_t total_kv_len = 0;
        for (int32_t i = 0; i < seq_cout; i++) {
            auto q_len = subsequence_begins.ptr<int32_t>()[i + 1] - subsequence_begins.ptr<int32_t>()[i];
            auto kv_len = past_lens.ptr<int32_t>()[i] + q_len;
            _score_offsets_aligned.ptr<int32_t>()[i] = total_kv_len_aligned;
            _score_offsets.ptr<int32_t>()[i] = total_kv_len;
            // aligned to cache line to avoid false sharing
            total_kv_len_aligned += rnd_up(kv_len, cache_line_size / sizeof(float));
            total_kv_len += kv_len;
        }

        _score_output.resize<float>({total_kv_len_aligned * H});
    }

    // compute one block(such as 32 tokens) of query in M dimension: softmax(q_block*k')*v
    // all tensors such as query... have no batch dimension because batch dimension is varying
    //  query: [H, L, S]
    //  present_value: [block_number, H, 32, S]
    //  output_emb: [L, H * S]
    //  qk_scratch_b: [rnd_up(kv_len, block_size), Hk, scratch_b_size]
    //  wv_scratch_b: [rnd_up(kv_len, block_size), Hk, scratch_b_size]
    void exec_kernel_multiple(const PlainTensor& query,
                              const PlainTensor& present_value,
                              const PlainTensor& output_emb,
                              const PlainTensor& qk_scratch_b,
                              const PlainTensor& wv_scratch_b,
                              const int32_t* block_table,
                              size_t ithr,
                              size_t q_blk,
                              size_t hq_beg,
                              size_t hq_end,
                              size_t hk,
                              size_t q_len,
                              size_t cur_kv_len,
                              const PlainTensor& alibi_slopes,
                              float* score_output) {
        auto q_start = q_blk * _block_size;
        auto q_end = std::min(q_start + _block_size, q_len);
        auto q_cnt = q_end - q_start;
        constexpr bool q_is_xf16 = one_of(precision_of<DATA_TYPE>::value, ov::element::bf16, ov::element::f16);
        constexpr bool q_cache_is_same = precision_of<DATA_TYPE>::value == VALUE_PREC;
        auto cur_kv_len_blocks = div_up(cur_kv_len, _block_size);
        for (size_t h = hq_beg; h < hq_end; h++) {
            auto* q_ptr = query.ptr<DATA_TYPE>(h, q_start, 0);
            auto* c_ptr = _weight.ptr<float>(ithr, h, 0, 0);
            // for each query block, loop through all key block
            // for blocks:
            // 1 0 0 0 ...
            // 1 1 0 0 ...
            // 1 1 1 0 ...
            // just computing the positions of 1 should be enough
            for (size_t k_blk = 0; k_blk < cur_kv_len_blocks; k_blk++) {
                auto* k_ptr = qk_scratch_b.ptr<DATA_TYPE>(k_blk, hk);
                _qk_gemm[q_cnt - 1]->executeGemm(q_cnt < _block_size,
                                                 q_ptr,
                                                 k_ptr,
                                                 c_ptr + k_blk * _block_size,
                                                 _wsp.data() + ithr * _wsp_size_per_thread,
                                                 _qk_scratch_a ? _qk_scratch_a.ptr<DATA_TYPE>(ithr, 0) : nullptr);
            }

            for (size_t m = q_start; m < q_end; m++) {
                // apply attention mask & sofmax
                auto ncausal = (cur_kv_len - q_cnt + (m - q_start) + 1);
                auto score = _weight.ptr<float>(ithr, h, m - q_start);
                if (_sliding_window) {
                    size_t start_idx = 0;
                    auto new_causal = ncausal;
                    float* alibi_lookup = nullptr;
                    if (ncausal > _sliding_window) {
                        start_idx = ncausal - static_cast<size_t>(_sliding_window);
                        new_causal = _sliding_window;
                    }
                    attn_softmax_kernel<float>(score + start_idx,
                                               reinterpret_cast<DATA_TYPE*>(score) + start_idx,
                                               _d_scale,
                                               alibi_lookup,
                                               nullptr,
                                               nullptr,
                                               false,
                                               new_causal,
                                               rnd_up(cur_kv_len, _block_size) - start_idx,
                                               precision_of<DATA_TYPE>::value,
                                               precision_of<DATA_TYPE>::value);

                    memset(score, 0, sizeof(DATA_TYPE) * start_idx);
                } else {
                    // alibi may available when _sliding_window is false
                    float* alibi_lookup = nullptr;
                    float alibi_slope = 0.f;
                    if (alibi_slopes) {
                        alibi_slope = alibi_slopes.ptr<float>()[h];
                        alibi_lookup = _alibi_lookup.ptr<float>() + _alibi_lookup.m_dims[0] - ncausal;
                    }
                    attn_softmax_kernel<float>(score,
                                               reinterpret_cast<DATA_TYPE*>(score),
                                               _d_scale,
                                               alibi_lookup,
                                               nullptr,
                                               nullptr,
                                               false,
                                               ncausal,
                                               rnd_up(cur_kv_len, _block_size),
                                               precision_of<DATA_TYPE>::value,
                                               precision_of<DATA_TYPE>::value,
                                               alibi_slope);
                }
                if (score_output) {
                    cvt_copy(score_output + h * rnd_up(cur_kv_len, 16),
                             reinterpret_cast<DATA_TYPE*>(score),
                             1,
                             cur_kv_len,
                             0,
                             0);
                }
            }

            // reuse float buffer, need to use float to compute offset
            auto* w_ptr = reinterpret_cast<DATA_TYPE*>(_weight.ptr<float>(ithr, h, 0, 0));
            float* fp32_out_ptr =
                q_is_xf16 ? _output.ptr<float>(ithr, 0, h, 0) : output_emb.ptr<float>(q_start, h * SV);

            // for each weight block, loop through all value block
            for (size_t v_blk = 0; v_blk < cur_kv_len_blocks; v_blk++) {
                DATA_TYPE* v_ptr;
                if (q_is_xf16 || !q_cache_is_same) {
                    v_ptr = wv_scratch_b.ptr<DATA_TYPE>(v_blk, hk);
                } else {
                    v_ptr = present_value.ptr<DATA_TYPE>(block_table[v_blk], hk);
                }
                if (v_blk == 0) {
                    _wv_gemm[q_cnt - 1]->executeGemm(q_cnt < _block_size,
                                                     w_ptr + v_blk * _block_size,
                                                     v_ptr,
                                                     fp32_out_ptr,
                                                     _wsp.data() + ithr * _wsp_size_per_thread,
                                                     _wv_scratch_a ? _wv_scratch_a.ptr<DATA_TYPE>(ithr, 0) : nullptr);
                } else {
                    _wv_gemm_acc[q_cnt - 1]->executeGemm(
                        q_cnt < _block_size,
                        w_ptr + v_blk * _block_size,
                        v_ptr,
                        fp32_out_ptr,
                        _wsp.data() + ithr * _wsp_size_per_thread,
                        _wv_scratch_a ? _wv_scratch_a.ptr<DATA_TYPE>(ithr, 0) : nullptr);
                }
            }
            if (q_is_xf16) {
                attn_memcpy2d_kernel(_output.ptr<float>(ithr, 0, h, 0),
                                     output_emb.ptr<DATA_TYPE>(q_start, h * SV),
                                     ov::element::f32,
                                     precision_of<DATA_TYPE>::value,
                                     _output.stride(1),
                                     output_emb.stride(0),
                                     SV,
                                     q_cnt);
            }
        }
    }

    // compute one token, loop along batch and head dimensions
    // all tensors such as query... have no batch dimension because batch dimension is varying
    //  query: [H, L, S]
    //  present_*: [block_number, H, 32, S]
    //  output_emb: [L, H * S]
    //  weight: [nthr, H, 32, rnd_up(kv_len, block_size)]
    //  output: [nthr, 32, H, S]
    void exec_kernel_one_bh(const PlainTensor& query,
                            const PlainTensor& present_key,
                            const PlainTensor& present_value,
                            const PlainTensor& output_emb,
                            const int32_t* block_table,
                            size_t ithr,
                            size_t hq_beg,
                            size_t hq_end,
                            size_t hk,
                            size_t q_len,
                            size_t cur_kv_len,
                            const PlainTensor& alibi_slopes,
                            float* score_output) {
#    if defined(OPENVINO_ARCH_X86_64)
        if (one_of(_fastpath_valid_prec, ov::element::bf16, ov::element::f16)) {
            _gemv->tile_config();
            for (size_t pk = 0, i = 0; pk < cur_kv_len; pk += _block_size, i++) {
                auto block_number = block_table[i];
                for (size_t pq = 0; pq < q_len; pq++) {
                    for (size_t h = hq_beg; h < hq_end; h++) {
                        (*_gemv)(query.ptr<DATA_TYPE>(h, pq),
                                 present_key.ptr<KEY_CACHE_TYPE>(block_number, hk),
                                 _weight.ptr<float>(ithr, h, pq) + pk);
                    }
                }
            }
            _gemv->tile_release();
        } else {
#    endif
            for (size_t pk = 0, i = 0; pk < cur_kv_len; pk += _block_size, i++) {
                auto block_number = block_table[i];
                for (size_t pq = 0; pq < q_len; pq++) {
                    for (size_t h = hq_beg; h < hq_end; h++) {
                        if (precision_of<KEY_CACHE_TYPE>::value == ov::element::u8 && _quant_key_bychannel) {
                            dot_product_block_by_channel(query.ptr<DATA_TYPE>(h, pq),
                                                         present_key.ptr<uint8_t>(block_number, hk),
                                                         _weight.ptr<float>(ithr, h, pq) + pk,
                                                         S,
                                                         std::min(_block_size, cur_kv_len - pk));
                        } else {
                            dot_product_block(query.ptr<DATA_TYPE>(h, pq),
                                              present_key.ptr<KEY_CACHE_TYPE>(block_number, hk),
                                              _weight.ptr<float>(ithr, h, pq) + pk,
                                              S,
                                              std::min(_block_size, cur_kv_len - pk),
                                              _key_group_size);
                        }
                    }
                }
            }
#    if defined(OPENVINO_ARCH_X86_64)
        }
#    endif

        for (size_t pq = 0; pq < q_len; pq++) {
            for (size_t h = hq_beg; h < hq_end; h++) {
                // apply attention mask & sofmax
                float* alibi_lookup = nullptr;
                float alibi_slope = 0.f;
                if (alibi_slopes) {
                    alibi_slope = alibi_slopes.ptr<float>()[h];
                    alibi_lookup = _alibi_lookup.ptr<float>() + _alibi_lookup.m_dims[0] - cur_kv_len;
                }
                attn_softmax_kernel<float>(_weight.ptr<float>(ithr, h, pq),
                                           _weight.ptr<float>(ithr, h, pq),
                                           _d_scale,
                                           alibi_lookup,
                                           nullptr,
                                           nullptr,
                                           false,
                                           cur_kv_len,
                                           cur_kv_len,
                                           ov::element::f32,
                                           ov::element::f32,
                                           alibi_slope);
                if (score_output) {
                    memcpy(score_output + h * rnd_up(cur_kv_len, 16),
                           _weight.ptr<float>(ithr, h, pq),
                           cur_kv_len * sizeof(float));
                }
            }
        }

        memset(_output.ptr<float>(ithr), 0, q_len * H * SV * sizeof(float));
        for (size_t pv = 0, i = 0; pv < cur_kv_len; pv += _block_size, i++) {
            auto block_number = block_table[i];
            for (size_t pq = 0; pq < q_len; pq++) {
                for (size_t h = hq_beg; h < hq_end; h++) {
                    auto sub_byte_multiplier = get_sub_byte_multiplier(present_value.get_precision());
                    size_t v_stride = (block_number * present_value.m_strides[0] + hk * present_value.m_strides[1]) *
                                      present_value.get_precision().size() / sub_byte_multiplier;
                    auto* v_ptr = reinterpret_cast<typename element_type_traits<VALUE_PREC>::value_type*>(
                        present_value.m_ptr.get() + v_stride);
                    attn_acc_value_block<typename element_type_traits<VALUE_PREC>::value_type, VALUE_PREC>(
                        _output.ptr<float>(ithr, pq, h),
                        _weight.ptr<float>(ithr, h, pq) + pv,
                        v_ptr,
                        SV,
                        std::min(_block_size, cur_kv_len - pv),
                        _value_group_size);
                }
            }
        }
        // convert to dst
        for (size_t pq = 0; pq < q_len; pq++) {
            for (size_t h = hq_beg; h < hq_end; h++) {
                cvt_copy(output_emb.ptr<DATA_TYPE>(pq, h * SV), _output.ptr<float>(ithr, pq, h), 1, SV, 0, 0);
            }
        }
    }

    // compute one token, loop along batch, head dimensions and kv_len, it's special for very long kv_len with small
    // batch tokens. It will assume NO mixture execution of first and second token. all tensors such as query... have
    // batch dimension which is DIFFERENT from above
    //  query: [B, H, L, S]
    //  key_cache: [block_number, H, _block_size, S]
    //  value_cache: [block_number, H, _block_size, Sv]
    //  output_emb: [B, L, H * S]
    // 3 loops along batch, head, kv cache length dimensions
    void exec_loop_bhl(const PlainTensor& query,
                       PlainTensor& key_cache,
                       PlainTensor& value_cache,
                       const PlainTensor& output_emb,
                       const PlainTensor& output_score,
                       size_t max_context_len,
                       const PlainTensor& past_lens,
                       const PlainTensor& subsequence_begins,
                       const PlainTensor& block_indices,
                       const PlainTensor& block_indices_begins,
                       const PlainTensor& alibi_slopes) {
        auto B = past_lens.size(0);
        auto q_len = query.size(2);
        auto kv_len_in_blocks = div_up(max_context_len, _block_size);
        // aligned to cache line (64bytes=16*sizeof(float)) to avoid false sharing
        _weight_bhl.resize<float>({B, H, q_len, rnd_up(max_context_len, std::max(_block_size, size_t{16}))});

        // for small batches dynamic scheduler has notable overhead
        bool prefer_static_loop;
        // if less than 2 work items per thread, loop H
        bool loop_hk = B * kv_len_in_blocks * Hk <= 2 * _nthr ? false : true;
        if (B <= 32) {
            prefer_static_loop = true;
            // small batch and all batch size is same(like SDPA case)
            auto kv_len = past_lens.ptr<int32_t>()[0];
            for (size_t b = 1; b < B; b++) {
                if (past_lens.ptr<int32_t>()[b] != kv_len) {
                    prefer_static_loop = false;
                }
            }
        } else {
            // for bigger batch skip the test to save the cost
            prefer_static_loop = false;
        }
        auto get_h_params =
            [](bool loop_hk, size_t hx, size_t h_each_group_len, size_t& hq_beg, size_t& hq_end, size_t& hk) {
                if (loop_hk) {
                    hk = hx;
                    hq_beg = hk * h_each_group_len;
                    hq_end = (hk + 1) * h_each_group_len;
                } else {
                    hq_beg = hx;
                    hq_end = hx + 1;
                    hk = hx / h_each_group_len;
                }
            };
        auto loop_qk = [&](size_t b, size_t pk_in_blocks, size_t hx) {
            auto context_len = static_cast<size_t>(past_lens.ptr<int32_t>()[b]) + 1;
            size_t hk, hq_beg, hq_end;
            get_h_params(loop_hk, hx, _h_each_group_len, hq_beg, hq_end, hk);

            // kv_len must be valid
            auto pk = pk_in_blocks * _block_size;
            if (pk < context_len) {
                auto block_number = block_indices.ptr<int32_t>()[block_indices_begins.ptr<int32_t>()[b] + pk_in_blocks];
#    if defined(OPENVINO_ARCH_X86_64)
                if (one_of(_fastpath_valid_prec, ov::element::bf16, ov::element::f16)) {
                    _gemv->tile_config();
                    for (size_t pq = 0; pq < q_len; pq++) {
                        for (size_t h = hq_beg; h < hq_end; h++) {
                            (*_gemv)(query.ptr<DATA_TYPE>(b, h, pq),
                                     key_cache.ptr<KEY_CACHE_TYPE>(block_number, hk),
                                     _weight_bhl.ptr<float>(b, h, pq) + pk);
                        }
                    }
                    _gemv->tile_release();
                } else {
#    endif
                    for (size_t pq = 0; pq < q_len; pq++) {
                        for (size_t h = hq_beg; h < hq_end; h++) {
                            if (precision_of<KEY_CACHE_TYPE>::value == ov::element::u8 && _quant_key_bychannel) {
                                dot_product_block_by_channel(query.ptr<DATA_TYPE>(b, h, pq),
                                                             key_cache.ptr<uint8_t>(block_number, hk),
                                                             _weight_bhl.ptr<float>(b, h, pq) + pk,
                                                             S,
                                                             std::min(_block_size, context_len - pk));
                            } else {
                                dot_product_block(query.ptr<DATA_TYPE>(b, h, pq),
                                                  key_cache.ptr<KEY_CACHE_TYPE>(block_number, hk),
                                                  _weight_bhl.ptr<float>(b, h, pq) + pk,
                                                  S,
                                                  std::min(_block_size, context_len - pk),
                                                  _key_group_size);
                            }
                        }
                    }
#    if defined(OPENVINO_ARCH_X86_64)
                }
#    endif
            }
        };

        auto loop_softmax = [&](size_t b, size_t h, size_t pq) {
            auto cur_kv_len = static_cast<size_t>(past_lens.ptr<int32_t>()[b]) + 1;
            auto ncausal = cur_kv_len;
            // apply attention mask & sofmax
            float* alibi_lookup = nullptr;
            float alibi_slope = 0.f;
            if (alibi_slopes) {
                alibi_slope = alibi_slopes.ptr<float>()[h];
                alibi_lookup = _alibi_lookup.ptr<float>() + _alibi_lookup.m_dims[0] - cur_kv_len;
            }
            attn_softmax_kernel<float>(_weight_bhl.ptr<float>(b, h, pq),
                                       _weight_bhl.ptr<float>(b, h, pq),
                                       _d_scale,
                                       alibi_lookup,
                                       nullptr,
                                       nullptr,
                                       false,
                                       ncausal,
                                       cur_kv_len,
                                       ov::element::f32,
                                       ov::element::f32,
                                       alibi_slope);
        };

        size_t h_dims = loop_hk ? Hk : H;
        if (prefer_static_loop) {
            parallel_for3d(B, kv_len_in_blocks, h_dims, loop_qk);
            parallel_for3d(B, H, q_len, loop_softmax);
        } else {
            parallel_for3d_dynamic(B, kv_len_in_blocks, h_dims, loop_qk);
            parallel_for3d_dynamic(B, H, q_len, loop_softmax);
        }

        if (output_score) {
            parallel_for2d_dynamic(B, q_len, [&](size_t b, size_t pq) {
                auto cur_kv_len = static_cast<size_t>(past_lens.ptr<int32_t>()[b]) + 1;
                auto* src = _weight_bhl.ptr<float>(b, 0, pq);
                size_t src_stride = _weight_bhl.stride(2);
                auto* dst = output_score.ptr<float>() + _score_offsets.ptr<int32_t>()[b];
                attn_reduce(dst, src, H, cur_kv_len, src_stride);
            });
        }

        // attn_w * V
        _output_bhl.resize<float>({static_cast<size_t>(_nthr), B, q_len, H, SV});
        // m_attn_w {B, H, q_len, kv_len}
        parallel_nt_static(_nthr, [&](const size_t ithr, const size_t nthr) {
            memset(_output_bhl.ptr<float>(ithr, 0, 0, 0, 0), 0, _output_bhl.stride(0) * sizeof(float));
        });

        auto loop_wk = [&](size_t b, size_t pv_in_blocks, size_t hx) {
            auto ithr = parallel_get_thread_num();
            auto context_len = static_cast<size_t>(past_lens.ptr<int32_t>()[b]) + 1;
            auto pv = pv_in_blocks * _block_size;
            size_t hk, hq_beg, hq_end;
            get_h_params(loop_hk, hx, _h_each_group_len, hq_beg, hq_end, hk);

            // kv_len must be valid
            if (pv < context_len) {
                auto block_number = block_indices.ptr<int32_t>()[block_indices_begins.ptr<int32_t>()[b] + pv_in_blocks];
                for (size_t pq = 0; pq < q_len; pq++) {
                    for (size_t h = hq_beg; h < hq_end; h++) {
                        auto sub_byte_multiplier = get_sub_byte_multiplier(value_cache.get_precision());
                        size_t v_stride = (block_number * value_cache.m_strides[0] + hk * value_cache.m_strides[1]) *
                                          value_cache.get_precision().size() / sub_byte_multiplier;
                        auto* v_ptr = reinterpret_cast<typename element_type_traits<VALUE_PREC>::value_type*>(
                            value_cache.m_ptr.get() + v_stride);
                        attn_acc_value_block<typename element_type_traits<VALUE_PREC>::value_type, VALUE_PREC>(
                            _output_bhl.ptr<float>(ithr, b, pq, h),
                            _weight_bhl.ptr<float>(b, h, pq) + pv,
                            v_ptr,
                            SV,
                            std::min(_block_size, context_len - pv),
                            _value_group_size);
                    }
                }
            }
        };

        if (prefer_static_loop) {
            parallel_for3d(B, kv_len_in_blocks, loop_hk ? Hk : H, loop_wk);
        } else {
            parallel_for3d_dynamic(B, kv_len_in_blocks, loop_hk ? Hk : H, loop_wk);
        }

        parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
            auto* temp = _output_bhl.ptr<float>(0, b, pq, h);
            size_t temp_stride = _output_bhl.stride(0);
            auto* dst = output_emb.ptr<DATA_TYPE>(b, pq, h * SV);
            attn_reduce(dst, temp, _nthr, SV, temp_stride);
        });
    }
};

template <typename DATA_TYPE, typename KEY_CACHE_TYPE, ov::element::Type_t VALUE_PREC>
struct MHA {
    MHAHelper<DATA_TYPE, KEY_CACHE_TYPE, VALUE_PREC>& _helper;
    struct AttnWorkItem {
        int32_t batch_in_reorder;  // which batch in reorder buffer will be used
        int32_t batch_in_seq;      // batch idx in sequence
        int32_t q_len;             // current sequence length, 1 for second token, 2+ for first token
        int32_t q_block_id;        // block id in this seq, valid at first token
    };
    struct ReorderWorkItem {
        int32_t batch_in_seq;      // batch idx in sequence
        int32_t batch_in_reorder;  // which batch in reorder buffer will be used
        int32_t kv_block_id;       // block id in this kv cache seq
    };
    struct WorkItems {
    private:
        std::vector<AttnWorkItem> attn_items;
        std::vector<ReorderWorkItem> reorder_items;
        int32_t max_kv_len_in_reorder;  // max kv len between first tokens
        int32_t max_batch_in_reorder;
        int32_t total_kv_len;

    public:
        void reset(const PlainTensor& query,
                   const PlainTensor& past_lens,
                   const PlainTensor& subsequence_begins,
                   size_t block_size) {
            attn_items.clear();
            reorder_items.clear();
            max_kv_len_in_reorder = 0;
            max_batch_in_reorder = 0;
            total_kv_len = 0;

            auto seq_cout = static_cast<int32_t>(past_lens.m_dims[0]);
            for (int32_t i = 0; i < seq_cout; i++) {
                auto q_len = subsequence_begins.ptr<int32_t>()[i + 1] - subsequence_begins.ptr<int32_t>()[i];
                auto kv_len = past_lens.ptr<int32_t>()[i] + q_len;
                auto kv_len_in_block = static_cast<int32_t>(div_up(kv_len, block_size));
                if (q_len == 1) {
                    attn_items.emplace_back(AttnWorkItem{0,     // batch_in_reorder
                                                         i,     // batch_in_seq
                                                         1ull,  // q_len
                                                         // kv_len in blocks, used in the sort function
                                                         kv_len_in_block - 1});
                } else {
                    auto reorder_sub_work_count = kv_len_in_block;
                    max_kv_len_in_reorder = std::max(max_kv_len_in_reorder, kv_len);
                    for (int32_t block_id = 0; block_id < reorder_sub_work_count; block_id++) {
                        reorder_items.emplace_back(ReorderWorkItem{
                            i,                     // batch_in_seq
                            max_batch_in_reorder,  // batch_in_reorder
                            block_id               // kv_block_id
                        });
                    }

                    // workitems for attention
                    auto attn_sub_work_count = static_cast<int32_t>(div_up(q_len, block_size));
                    for (int32_t block_id = 0; block_id < attn_sub_work_count; block_id++) {
                        attn_items.emplace_back(AttnWorkItem{
                            max_batch_in_reorder,  // batch_in_reorder
                            i,                     // batch_in_seq
                            q_len,                 // q_len
                            block_id               // q_block_id
                        });
                    }
                    max_batch_in_reorder++;
                }
                total_kv_len += kv_len;
            }
            // std::sort(attn_items.begin(), attn_items.end(), [] (const AttnWorkItem& left, const AttnWorkItem& right)
            // {
            //     // kv block number which will be acessed later
            //     auto left_kv_blocks = left.q_block_id;
            //     auto right_kv_blocks = right.q_block_id;
            //     return left_kv_blocks > right_kv_blocks;
            // });
        }
        [[nodiscard]] const AttnWorkItem& get_attn_work_item(size_t idx) const {
            return attn_items[idx];
        }
        [[nodiscard]] size_t attn_work_size() const {
            return attn_items.size();
        }
        [[nodiscard]] const ReorderWorkItem& get_reorder_work_item(size_t idx) const {
            return reorder_items[idx];
        }
        [[nodiscard]] size_t reorder_work_size() const {
            return reorder_items.size();
        }
        [[nodiscard]] size_t get_reorder_max_batch_size() const {
            return static_cast<size_t>(max_batch_in_reorder);
        }
        [[nodiscard]] size_t get_reorder_max_kv_len() const {
            return static_cast<size_t>(max_kv_len_in_reorder);
        }
        [[nodiscard]] size_t get_total_kv_len() const {
            return static_cast<size_t>(total_kv_len);
        }
    };

    WorkItems _workitems;

    MHA(MHAHelper<DATA_TYPE, KEY_CACHE_TYPE, VALUE_PREC>& helper) : _helper(helper) {}

    // one loop to handle first and second tokens
    void exec_loop_mixed(const PlainTensor& q,
                         PlainTensor& k_cache,
                         const PlainTensor& v_cache,
                         const PlainTensor& output_emb,
                         const PlainTensor& output_score,
                         size_t max_context_len,
                         const PlainTensor& past_lens,
                         const PlainTensor& subsequence_begins,
                         const PlainTensor& block_indices,
                         const PlainTensor& block_indices_begins,
                         const PlainTensor& alibi_slopes) {
        auto Hk = v_cache.m_dims[1];

        constexpr bool q_is_xf16 = one_of(precision_of<DATA_TYPE>::value, ov::element::bf16, ov::element::f16);
        constexpr bool q_cache_is_same = precision_of<DATA_TYPE>::value == VALUE_PREC;
        auto attn_work_count = _workitems.attn_work_size();
        auto reorder_work_count = _workitems.reorder_work_size();

        // buffer for transpose and repack
        _helper.init_reorder_buffers(_workitems.get_reorder_max_batch_size(),
                                     div_up(_workitems.get_reorder_max_kv_len(), _helper._block_size));

        // packed k, v
        parallel_for2d_dynamic(reorder_work_count, Hk, [&](size_t w, size_t hk) {
            const auto& item = _workitems.get_reorder_work_item(w);
            const auto batch_in_seq = item.batch_in_seq;
            const auto batch_in_reorder = item.batch_in_reorder;
            const auto kv_block = item.kv_block_id;
            auto block_number =
                block_indices.ptr<int32_t>()[block_indices_begins.ptr<int32_t>()[batch_in_seq] + kv_block];
            if (block_number < 0) {
                return;
            }

            auto ithr = parallel_get_thread_num();
            auto* k_ptr = k_cache.ptr<KEY_CACHE_TYPE>(block_number, hk);

            transpose_16NxK<DATA_TYPE, precision_of<KEY_CACHE_TYPE>::value>(
                _helper._qk_scratch_b.template ptr<DATA_TYPE>(batch_in_reorder, kv_block, hk),
                k_ptr,
                _helper._output.template ptr<DATA_TYPE>(ithr),
                _helper._block_size,            // N
                _helper.S,                      // K
                _helper._block_size,            // dst_stride
                _helper.S,                      // src_stride
                _helper._key_group_size,        // group_size
                _helper._quant_key_bychannel);  // quant_by_channel

            if (q_is_xf16) {
                auto sub_byte_multiplier = get_sub_byte_multiplier(v_cache.get_precision());
                size_t v_stride = (block_number * v_cache.m_strides[0] + hk * v_cache.m_strides[1]) *
                                  v_cache.get_precision().size() / sub_byte_multiplier;
                auto* v_ptr = v_cache.m_ptr.get() + v_stride;
                pack_32NxK<DATA_TYPE, VALUE_PREC>(
                    _helper._wv_scratch_b.template ptr<DATA_TYPE>(batch_in_reorder, kv_block, hk),
                    v_ptr,
                    _helper._output.template ptr<DATA_TYPE>(ithr),
                    _helper._block_size,
                    _helper.SV,
                    rnd_up(_helper.SV, _helper._block_size),
                    _helper.SV,
                    _helper._value_group_size);
            } else {
                // need to decompress
                if (!q_cache_is_same) {
                    auto sub_byte_multiplier = get_sub_byte_multiplier(v_cache.get_precision());
                    size_t v_stride = (block_number * v_cache.m_strides[0] + hk * v_cache.m_strides[1]) *
                                      v_cache.get_precision().size() / sub_byte_multiplier;
                    auto* v_ptr = v_cache.m_ptr.get() + v_stride;
                    dequant<DATA_TYPE, VALUE_PREC>(
                        _helper._wv_scratch_b.template ptr<DATA_TYPE>(batch_in_reorder, kv_block, hk),
                        v_ptr,
                        _helper._block_size,
                        _helper.SV,
                        _helper._value_group_size);
                }
            }
        });

        // loop along HK dimension: if mixed first/second token and elements count is enough, loop HK to reuse KV in the
        // CPU cache
        //    else if elements count is small, prefer to loop H to get more work to avoid thread imbalance
        bool loop_hk = _workitems.get_reorder_max_batch_size() == past_lens.m_dims[0] ||  // if only first token, loop H
                               attn_work_count * Hk <= 2 * _helper._nthr
                           ? false
                           : true;  // or less than 2 work items per thread, loop H

        parallel_for2d_dynamic(attn_work_count, loop_hk ? Hk : _helper.H, [&](size_t w, size_t hx) {
            size_t hk, hq_beg, hq_end;
            if (loop_hk) {
                hk = hx;
                hq_beg = hk * _helper._h_each_group_len;
                hq_end = (hk + 1) * _helper._h_each_group_len;
            } else {
                hq_beg = hx;
                hq_end = hx + 1;
                hk = hx / _helper._h_each_group_len;
            }

            const auto& item = _workitems.get_attn_work_item(w);
            const auto batch_in_seq = item.batch_in_seq;
            const auto batch_in_token = subsequence_begins.ptr<int32_t>()[batch_in_seq];
            const auto q_len = static_cast<size_t>(item.q_len);
            size_t ithr = parallel_get_thread_num();

            if (q_len == 1) {
                const auto cur_kv_len = static_cast<size_t>(past_lens.ptr<int32_t>()[batch_in_seq]) + 1;
                float* score_output = nullptr;
                if (output_score) {
                    auto score_offset = _helper._score_offsets_aligned.template ptr<int32_t>()[batch_in_seq];
                    score_output = _helper._score_output.template ptr<float>() + score_offset * _helper.H;
                }

                _helper.exec_kernel_one_bh(
                    q.slice(0, batch_in_token, batch_in_token),
                    k_cache,
                    v_cache,
                    output_emb.slice(0, batch_in_token, batch_in_token),
                    block_indices.ptr<int32_t>() + block_indices_begins.ptr<int32_t>()[batch_in_seq],
                    ithr,
                    hq_beg,
                    hq_end,
                    hk,
                    1ul,
                    cur_kv_len,
                    alibi_slopes,
                    score_output);
            } else {
                const auto batch_in_reorder = item.batch_in_reorder;
                const auto q_blk = item.q_block_id;
                const auto q_cnt = std::min(_helper._block_size, q_len - q_blk * _helper._block_size);
                const auto cur_kv_len =
                    static_cast<size_t>(past_lens.ptr<int32_t>()[batch_in_seq]) + q_blk * _helper._block_size + q_cnt;
                float* score_output = nullptr;
                if (output_score) {
                    // last block
                    if (q_len - q_blk * _helper._block_size <= _helper._block_size) {
                        auto score_offset = _helper._score_offsets_aligned.template ptr<int32_t>()[batch_in_seq];
                        score_output = _helper._score_output.template ptr<float>() + score_offset * _helper.H;
                    }
                }

                PlainTensor sub_query;
                sub_query.resize({q_len, _helper.H, _helper.S}, q.ptr<DATA_TYPE>(batch_in_token));
                sub_query = sub_query.permute({1, 0, 2});
                _helper.exec_kernel_multiple(
                    sub_query,
                    v_cache,
                    output_emb.slice(0, batch_in_token, batch_in_token + q_len)
                        .reshape({q_len, _helper.H * _helper.SV}),
                    _helper._qk_scratch_b.slice(0, batch_in_reorder, batch_in_reorder),
                    _helper._wv_scratch_b.slice(0, batch_in_reorder, batch_in_reorder),
                    block_indices.ptr<int32_t>() + block_indices_begins.ptr<int32_t>()[batch_in_seq],
                    ithr,
                    q_blk,
                    hq_beg,
                    hq_end,
                    hk,
                    q_len,
                    cur_kv_len,
                    alibi_slopes,
                    score_output);
            }
        });
        if (output_score) {
            parallel_for2d_dynamic(past_lens.m_dims[0], 1, [&](size_t b, size_t pq) {
                auto seq_len = static_cast<size_t>(subsequence_begins.ptr<int32_t>()[b + 1] -
                                                   subsequence_begins.ptr<int32_t>()[b]);
                auto cur_kv_len = static_cast<size_t>(past_lens.ptr<int32_t>()[b]) + seq_len;
                auto src_offset = _helper._score_offsets_aligned.template ptr<int32_t>()[b];
                auto* src = _helper._score_output.template ptr<float>() + src_offset * _helper.H;
                size_t src_stride = rnd_up(cur_kv_len, 16);
                auto dst_offset = _helper._score_offsets.template ptr<int32_t>()[b];
                auto* dst = output_score.ptr<float>() + dst_offset;
                attn_reduce(dst, src, _helper.H, cur_kv_len, src_stride);
            });
        }
    }

    // Q, K, V is ready, do attention
    void operator()(PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    PlainTensor& output_emb,
                    PlainTensor& output_score,
                    size_t max_context_len,
                    const PlainTensor& past_lens,
                    const PlainTensor& subsequence_begins,
                    const PlainTensor& block_indices,
                    const PlainTensor& block_indices_begins,
                    const PlainTensor& alibi_slopes) {
        _workitems.reset(query, past_lens, subsequence_begins, _helper._block_size);
        if (output_score) {
            _helper.init_score_buffers(past_lens, subsequence_begins);
        }

        auto nthr = static_cast<size_t>(parallel_get_max_threads());

        if (past_lens.m_dims[0] >= nthr || _workitems.get_reorder_max_batch_size() > 0) {
            exec_loop_mixed(query,
                            present_key,
                            present_value,
                            output_emb,
                            output_score,
                            max_context_len,
                            past_lens,
                            subsequence_begins,
                            block_indices,
                            block_indices_begins,
                            alibi_slopes);
        } else {
            _helper.exec_loop_bhl(query,
                                  present_key,
                                  present_value,
                                  output_emb,
                                  output_score,
                                  max_context_len,
                                  past_lens,
                                  subsequence_begins,
                                  block_indices,
                                  block_indices_begins,
                                  alibi_slopes);
        }
    }
};

template <typename DATA_TYPE, typename KEY_CACHE_TYPE, ov::element::Type_t VALUE_PREC>
struct AttentionExecutor : public PagedAttentionExecutor {
    MHAHelper<DATA_TYPE, KEY_CACHE_TYPE, VALUE_PREC> _helper;
    MHA<DATA_TYPE, KEY_CACHE_TYPE, VALUE_PREC> _kernel;
    PlainTensor _slot_mapping;

    AttentionExecutor() : _kernel(_helper) {}

    explicit AttentionExecutor(size_t key_group_size, size_t value_group_size, bool quant_key_bychannel)
        : _helper(
              MHAHelper<DATA_TYPE, KEY_CACHE_TYPE, VALUE_PREC>(key_group_size, value_group_size, quant_key_bychannel)),
          _kernel(_helper) {}

    void init(const std::vector<MemoryPtr>& inputs,
              const std::vector<MemoryPtr>& outputs,
              PlainTensor& q,
              PlainTensor& k,
              PlainTensor& v,
              PlainTensor& k_cache,
              PlainTensor& v_cache,
              PlainTensor& past_lens,
              PlainTensor& subsequence_begins,
              PlainTensor& block_indices,
              PlainTensor& block_indices_begins,
              float& scale,
              size_t& sliding_window,
              PlainTensor& alibi_slopes,
              size_t& max_context_len,
              PlainTensor& rotated_block_indices,
              PlainTensor& rotation_deltas,
              PlainTensor& rotation_trig_lut,
              PlainTensor& output_emb,
              PlainTensor& output_score) {
        q.reset(inputs[ID_Q]);  // [B_token, H * S]
        k.reset(inputs[ID_K]);
        v.reset(inputs[ID_V]);
        k_cache.reset(inputs[ID_KCACHE]);                             // [NUM_BLOCKS, H, 32, S]
        v_cache.reset(inputs[ID_VCACHE]);                             // [NUM_BLOCKS, H, 32, S]
        past_lens.reset(inputs[ID_PAST_LENS]);                        // [B_seq]
        subsequence_begins.reset(inputs[ID_SUBSEQUENCE_BEGINS]);      // [B_seq+1]
        block_indices.reset(inputs[ID_BLOCK_INDICES]);                // [num_blocks]
        block_indices_begins.reset(inputs[ID_BLOCK_INDICES_BEGINS]);  // [B_seq+1]
        scale = *inputs[ID_SCALE]->getDataAs<float>();
        sliding_window = static_cast<size_t>(*inputs[ID_SLIDING_WINDOW]->getDataAs<int32_t>());
        if (!inputs[ID_ALIBI_SLOPES]->getShape().hasZeroDims()) {
            alibi_slopes.reset(inputs[ID_ALIBI_SLOPES]);
        }
        max_context_len = static_cast<size_t>(*inputs[ID_MAX_CONTEXT_LEN]->getDataAs<int32_t>());

        size_t inputs_size = inputs.size();
        if (inputs_size > ID_ROTATED_BLOCK_INDICES) {
            OPENVINO_ASSERT(inputs_size >= ID_ROTATION_TRIG_LUT);
            if (!inputs[ID_ROTATED_BLOCK_INDICES]->getShape().hasZeroDims()) {
                rotated_block_indices.reset(inputs[ID_ROTATED_BLOCK_INDICES]);  // [num_blocks]
            }
            if (!inputs[ID_ROTATION_DELTAS]->getShape().hasZeroDims()) {
                rotation_deltas.reset(inputs[ID_ROTATION_DELTAS]);  // [num_blocks,  block_size (32) || 1]
            }
            if (!inputs[ID_ROTATION_TRIG_LUT]->getShape().hasZeroDims()) {
                rotation_trig_lut.reset(
                    inputs[ID_ROTATION_TRIG_LUT]);  // [max_context_len * embedding_size], row-major layout
            }
        }

        output_emb.reset(outputs[0]);
        if (outputs.size() == 2) {
            output_score.reset(outputs[1]);
        }

        auto B_token = q.size(0);
        auto Hk = k_cache.size(1);
        /* The layout for kv cache:

           by-hidden, quantized by S(hidden dims) group_num = N
           N * f32(scale + zp)|group_0|group_1|...|group_N
           adjusted_S = S + N * f32(scale + zp) * sub_byte_multiplier

           by channel, quantized by channel [block_size, S], group_num = block_size
           adjusted block consists 3 parts
           f32[scale, S]
           f32[zp, S]
           u8[block_size, S]
           adjusted key cache block u8[block_size + 2 * sizeof(float) * sub_byte_multiplier, S]
        */

        const size_t key_sub_byte_multiplier = get_sub_byte_multiplier(k_cache.get_precision());
        const size_t value_sub_byte_multiplier = get_sub_byte_multiplier(v_cache.get_precision());
        const size_t key_params_size = sizeof(float) * 2 * key_sub_byte_multiplier;
        // u4 needs scale + zp. s4 needs scale.
        const size_t param_size =
            one_of(v_cache.get_precision(), ov::element::u4, ov::element::u8) ? sizeof(float) * 2 : sizeof(float);
        const size_t value_params_size = param_size * value_sub_byte_multiplier;
        size_t key_group_num =
            _helper._key_group_size ? k_cache.size(3) / (_helper._key_group_size + key_params_size) : 1;
        size_t value_group_num =
            _helper._value_group_size ? v_cache.size(3) / (_helper._value_group_size + value_params_size) : 1;
        auto SV = v_cache.size(3) - (v_cache.get_precision().is_real() ? 0 : value_params_size * value_group_num);
        // revise group_size if it's zero.
        _helper._value_group_size = _helper._value_group_size ? _helper._value_group_size : SV;

        // check by_hidden_dims parameter of value cache
        if (!value_group_num && v_cache.get_precision().is_integral())
            OPENVINO_THROW("PagedAttn value cache gets wrong group_size, ",
                           _helper._value_group_size,
                           " should be smaller than hidden_dims");
        size_t S = 0;
        if (_helper._quant_key_bychannel) {
            S = k_cache.size(3);
        } else {
            // check by_hidden_dims parameter of key cache
            if (!key_group_num && k_cache.get_precision().is_integral())
                OPENVINO_THROW("PagedAttn key cache gets wrong group_size, ",
                               _helper._key_group_size,
                               " should be smaller than hidden_dims");
            S = k_cache.size(3) - (k_cache.get_precision().is_real() ? 0 : key_params_size * key_group_num);
            _helper._key_group_size = _helper._key_group_size ? _helper._key_group_size : S;
        }
        auto block_size = _helper._quant_key_bychannel ? (k_cache.size(2) - key_params_size) : k_cache.size(2);
        auto H = q.size(1) / S;
        auto h_each_group_len = 1;
        if (Hk != H) {
            h_each_group_len = H / Hk;
        }
        auto B_seq = past_lens.size(0);

        q.assert_dims({B_token, H * S});
        k.assert_dims({B_token, Hk * S});
        v.assert_dims({B_token, Hk * SV});
        q = q.reshape({B_token, H, 1, S});
        k = k.reshape({B_token, Hk, 1, S});
        v = v.reshape({B_token, Hk, 1, SV});

        if (k_cache.m_dt == ov::element::Type_t::u8) {
            if (_helper._quant_key_bychannel) {
                k_cache.assert_dims({0, Hk, block_size + key_params_size, S}, true);
            } else {
                k_cache.assert_dims({0, Hk, block_size, S + key_params_size * key_group_num}, true);
            }
            v_cache.assert_dims({k_cache.m_dims[0], Hk, block_size, SV + value_params_size * value_group_num});

        } else {
            k_cache.assert_dims({0, Hk, block_size, S}, true);
            v_cache.assert_dims({k_cache.m_dims[0], Hk, block_size, SV});
        }
        past_lens.assert_dims({B_seq});
        subsequence_begins.assert_dims({B_seq + 1});
        block_indices.assert_dims({0}, true);
        block_indices_begins.assert_dims({B_seq + 1});
        if (scale == 0.0f) {
            scale = 1.0f / sqrt(S);
        }
        if (alibi_slopes) {
            alibi_slopes.assert_dims({H});
        }

        bool init_rotation_coefficient_scratch = false;
        if (rotated_block_indices) {
            // Only K entries are needed to be rotated, since position is encoded at the Q^T @ (effective_RoPE_matrix) @
            // K matrix multiplication
            rotation_deltas.assert_dims({rotated_block_indices.size(0), 0}, /* special_zero = */ true);
            OPENVINO_ASSERT(rotation_deltas.shape()[1] == 1 ||
                            rotation_deltas.shape()[1] == block_size);  // per-block or per-token granularity
            rotation_trig_lut.assert_dims({0, S}, /* special_zero = */ true);
            init_rotation_coefficient_scratch = true;
        }
        output_emb.assert_dims({B_token, H * SV});
        output_emb = output_emb.reshape({B_token, 1, H * SV});

        // TODO: enable block_size to be multiple of 32
        OPENVINO_ASSERT(block_size == 32, "CPU: block size must be 32, current: ", block_size);

        _helper.init(H,
                     S,
                     SV,
                     Hk,
                     h_each_group_len,
                     block_size,
                     sliding_window,
                     scale,
                     max_context_len,
                     alibi_slopes,
                     init_rotation_coefficient_scratch);
    }

    void concat_pastkv(const PlainTensor& k,
                       const PlainTensor& v,
                       PlainTensor& k_cache,
                       PlainTensor& v_cache,
                       const PlainTensor& past_lens,
                       const PlainTensor& subsequence_begins,
                       const PlainTensor& block_indices,
                       const PlainTensor& block_indices_begins) {
        auto B_token = k.size(0);
        _slot_mapping.resize<int32_t>({B_token});

        size_t idx = 0;
        for (size_t i = 0; i < past_lens.size(0); i++) {
            auto q_len = subsequence_begins.ptr<int32_t>()[i + 1] - subsequence_begins.ptr<int32_t>()[i];
            auto kv_len = past_lens.ptr<int32_t>()[i] + q_len;
            auto block_number_start = block_indices_begins.ptr<int32_t>()[i];
            auto block_offset_start = kv_len - q_len;
            for (int32_t j = 0; j < q_len; j++) {
                auto block_offset = block_offset_start + j;
                auto block_number =
                    block_indices.ptr<int32_t>()[block_number_start + block_offset / _helper._block_size];
                _slot_mapping.ptr<int32_t>()[idx++] =
                    block_number * _helper._block_size + block_offset % _helper._block_size;
            }
            // To simplify tails of the kernels for Q*K and W*V:
            // for first token kernels:
            //    Q*K aka [m, k0] * [n0, k0]', there is already tails logic for m, k0, but no(always assume n0 is
            //    block_size) for n0 which means the tail of k_cache need to be set to zero.
            //    W*V aka [m, k1] * [n1, k1]', there is no tails handing for n1, so tails of v_cache need to be set to
            //    zero.
            // for second token, the kernels have tails handling logic
            if (q_len != 1 && kv_len % _helper._block_size != 0) {
                // block no. which contains tails
                auto block_number = block_indices.ptr<int32_t>()[block_number_start + kv_len / _helper._block_size];
                // tails start position
                auto block_offset = kv_len % _helper._block_size;
                // tails number
                auto zero_tokens = _helper._block_size - block_offset;
                auto S = k_cache.m_dims[3];
                auto SV = v_cache.m_dims[3];
                auto Hk = k_cache.m_dims[1];  // shape: [block, H, 32, S]
                parallel_for2d(Hk, zero_tokens, [&](size_t h, size_t l) {
                    auto set_zero =
                        [](PlainTensor& cache, size_t block_number, size_t h, size_t l, size_t hidden_dims) {
                            auto sub_byte_multiplier = get_sub_byte_multiplier(cache.get_precision());
                            size_t cache_stride =
                                (block_number * cache.stride(0) + h * cache.stride(1) + l * cache.stride(2)) *
                                cache.get_precision().size() / sub_byte_multiplier;
                            auto cache_ptr = cache.m_ptr.get() + cache_stride;
                            std::memset(cache_ptr, 0, hidden_dims * cache.m_element_size / sub_byte_multiplier);
                        };
                    set_zero(k_cache, block_number, h, block_offset + l, S);
                    set_zero(v_cache, block_number, h, block_offset + l, SV);
                });
            }
        }

        if (k_cache.m_dt == ov::element::Type_t::u8) {
            // slot_mapping could only be used for per token quantization
            // by_channel needs all data to calculation block info.
            paged_attn_quantkv(k,
                               v,
                               k_cache,
                               v_cache,
                               past_lens,
                               subsequence_begins,
                               block_indices,
                               block_indices_begins,
                               _slot_mapping,
                               _helper._output,
                               _helper._quant_key_bychannel,
                               _helper._key_group_size,
                               _helper._value_group_size);
        } else {
            paged_attn_memcpy(k, v, k_cache, v_cache, _slot_mapping);
        }
    }

    void execute(const std::vector<MemoryPtr>& inputs, const std::vector<MemoryPtr> outputs) override {
        PlainTensor q, k, v, k_cache, v_cache;
        PlainTensor past_lens, subsequence_begins, block_indices, block_indices_begins;
        float scale;
        size_t sliding_window;
        PlainTensor alibi_slopes;
        size_t max_context_len;
        PlainTensor rotated_block_indices;
        PlainTensor rotation_deltas;
        PlainTensor rotation_trig_lut;

        PlainTensor output_emb;
        PlainTensor output_score;

        init(inputs,
             outputs,
             q,
             k,
             v,
             k_cache,
             v_cache,
             past_lens,
             subsequence_begins,
             block_indices,
             block_indices_begins,
             scale,
             sliding_window,
             alibi_slopes,
             max_context_len,
             rotated_block_indices,
             rotation_deltas,
             rotation_trig_lut,
             output_emb,
             output_score);

        if (rotated_block_indices) {
            rotate_kv_cache<KEY_CACHE_TYPE>(k_cache,
                                            rotated_block_indices,
                                            rotation_deltas,
                                            rotation_trig_lut,
                                            _helper._block_rotation_coefficient_scratch);
        }

        concat_pastkv(k, v, k_cache, v_cache, past_lens, subsequence_begins, block_indices, block_indices_begins);

        _kernel(q,
                k_cache,
                v_cache,
                output_emb,
                output_score,
                max_context_len,
                past_lens,
                subsequence_begins,
                block_indices,
                block_indices_begins,
                alibi_slopes);
    }
};
#endif

std::shared_ptr<PagedAttentionExecutor> make_pa_executor(ov::element::Type data_type,
                                                         ov::element::Type key_cache_type,
                                                         ov::element::Type value_cache_type,
                                                         size_t key_group_size,
                                                         size_t value_group_size,
                                                         bool quant_key_bychannel) {
    std::shared_ptr<PagedAttentionExecutor> executor;
#if defined(OPENVINO_ARCH_X86_64)
    if (data_type == ov::element::bf16) {
#    if defined(HAVE_AVX512F)
        if (key_cache_type == ov::element::u8) {
            if (value_cache_type == ov::element::u4) {
                executor =
                    std::make_shared<AttentionExecutor<ov::bfloat16, uint8_t, ov::element::u4>>(key_group_size,
                                                                                                value_group_size,
                                                                                                quant_key_bychannel);
            } else if (value_cache_type == ov::element::u8) {
                executor =
                    std::make_shared<AttentionExecutor<ov::bfloat16, uint8_t, ov::element::u8>>(key_group_size,
                                                                                                value_group_size,
                                                                                                quant_key_bychannel);
            } else {
                OPENVINO_THROW("make_pa_executor: key_cache_type u8 with value_cache_type ",
                               value_cache_type.to_string(),
                               " is not support");
            }

        } else {
            OPENVINO_ASSERT(key_cache_type == ov::element::bf16, "expect kvcache type bf16, current: ", key_cache_type);
            executor = std::make_shared<AttentionExecutor<ov::bfloat16, ov::bfloat16, ov::element::bf16>>();
        }
#    else
        OPENVINO_THROW("make_pa_executor: bf16 needs avx512+ hardware.");
#    endif
    } else if (data_type == ov::element::f16) {
#    if defined(HAVE_AVX512F)
        if (key_cache_type == ov::element::u8) {
            if (value_cache_type == ov::element::u4) {
                executor =
                    std::make_shared<AttentionExecutor<ov::float16, uint8_t, ov::element::u4>>(key_group_size,
                                                                                               value_group_size,
                                                                                               quant_key_bychannel);
            } else if (value_cache_type == ov::element::u8) {
                executor =
                    std::make_shared<AttentionExecutor<ov::float16, uint8_t, ov::element::u8>>(key_group_size,
                                                                                               value_group_size,
                                                                                               quant_key_bychannel);
            } else {
                OPENVINO_THROW("make_pa_executor: key_cache_type u8 with value_cache_type ",
                               value_cache_type.to_string(),
                               " is not support");
            }
        } else {
            OPENVINO_ASSERT(key_cache_type == ov::element::f16, "expect kvcache type f16, current: ", key_cache_type);
            executor = std::make_shared<AttentionExecutor<ov::float16, ov::float16, ov::element::f16>>();
        }
#    else
        OPENVINO_THROW("make_pa_executor: f16 needs avx512+ hardware.");
#    endif
    } else if (data_type == ov::element::f32) {
        if (key_cache_type == ov::element::u8) {
            if (value_cache_type == ov::element::u4) {
                executor = std::make_shared<AttentionExecutor<float, uint8_t, ov::element::u4>>(key_group_size,
                                                                                                value_group_size,
                                                                                                quant_key_bychannel);
            } else if (value_cache_type == ov::element::u8) {
                executor = std::make_shared<AttentionExecutor<float, uint8_t, ov::element::u8>>(key_group_size,
                                                                                                value_group_size,
                                                                                                quant_key_bychannel);
            } else {
                OPENVINO_THROW("make_pa_executor: key_cache_type u8 with value_cache_type ",
                               value_cache_type.to_string(),
                               " is not support");
            }
        } else if (key_cache_type == ov::element::f16) {
            executor = std::make_shared<AttentionExecutor<float, ov::float16, ov::element::f16>>(key_group_size,
                                                                                                 value_group_size,
                                                                                                 quant_key_bychannel);
        } else {
            OPENVINO_ASSERT(key_cache_type == ov::element::f32, "expect kvcache type f32, current: ", key_cache_type);
            executor = std::make_shared<AttentionExecutor<float, float, ov::element::f32>>(key_group_size,
                                                                                           value_group_size,
                                                                                           quant_key_bychannel);
        }
    } else {
        OPENVINO_THROW("make_pa_executor: unsupported precision: ", data_type);
    }
#elif (defined(OPENVINO_ARCH_ARM64) && defined(HAVE_SVE))
    if (data_type == ov::element::f32) {
        if (key_cache_type == ov::element::u8 && value_cache_type == ov::element::u8) {
            executor = std::make_shared<AttentionExecutor<float, uint8_t, ov::element::u8>>(key_group_size,
                                                                                            value_group_size,
                                                                                            quant_key_bychannel);
        } else {
            OPENVINO_THROW("make_pa_executor: key_cache_type and value_cache_type of u8 is only support");
        }
    }

#else
    OPENVINO_THROW("make_pa_executor: only support x64 platform or ARM with SVE support");
#endif
    return executor;
}

}  // namespace ov::Extensions::Cpu::XARCH
