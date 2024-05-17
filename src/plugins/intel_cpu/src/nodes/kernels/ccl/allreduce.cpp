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

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/parallel.hpp"
#include "allreduce.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void allreduce_float32(const float* send_buf, float* recv_buf, size_t count) {
#if defined(HAVE_AVX512F)
    const size_t stride = 16;
    size_t step = count / stride;
    parallel_for(step, [&](size_t j){
      __m512 vecA, vecB, vecC;
      size_t i = j * stride;
      // iter 0
      vecA = _mm512_loadu_ps(send_buf + i);
      vecB = _mm512_loadu_ps(recv_buf + i);
      vecC = _mm512_add_ps(vecA, vecB);
      _mm512_storeu_ps(recv_buf + i, vecC);
      // // iter 1
      // __m512 vecX, vecY, vecZ;
      // size_t i1 = j * stride + stride;
      // vecX = _mm512_loadu_ps(send_buf + i1);
      // vecY = _mm512_loadu_ps(recv_buf + i1);
      // vecZ = _mm512_add_ps(vecX, vecY);
      // _mm512_storeu_ps(recv_buf + i1, vecZ);
    });
    size_t tail = count & ~(stride - 1);
    for (size_t i = tail; i < count; ++i) {
      recv_buf[i] += send_buf[i];
    }
#elif defined(HAVE_AVX2)
    const size_t stride = 8;
    size_t step = count / stride;
    parallel_for(step, [&](size_t j){
      __m256 vecA, vecB, vecC;
      size_t i = j * stride;
      vecA = _mm256_loadu_ps(send_buf + i);
      vecB = _mm256_loadu_ps(recv_buf + i);
      vecC = _mm256_add_ps(vecA, vecB);
      _mm256_storeu_ps(recv_buf + i, vecC);
    });
    size_t tail = count & ~(stride - 1);
    for (size_t i = tail; i < count; ++i) {
        recv_buf[i] += send_buf[i];
    }
#else
    const size_t stride = 8;
    const size_t unloop = 8;
    size_t step = count / unloop;
    parallel_for(step, [&](size_t i) {
      recv_buf[i * unloop]     += send_buf[i * unloop];
      recv_buf[i * unloop + 1] += send_buf[i * unloop + 1];
      recv_buf[i * unloop + 2] += send_buf[i * unloop + 2];
      recv_buf[i * unloop + 3] += send_buf[i * unloop + 3];
      recv_buf[i * unloop + 4] += send_buf[i * unloop + 4];
      recv_buf[i * unloop + 5] += send_buf[i * unloop + 5];
      recv_buf[i * unloop + 6] += send_buf[i * unloop + 6];
      recv_buf[i * unloop + 7] += send_buf[i * unloop + 7];
    });
    size_t tail = count & ~(stride - 1);
    for (size_t i = tail; i < count; ++i) {
      recv_buf[i] += send_buf[i];
    }
#endif
}

#if defined(HAVE_AVX512F)
__m512 cvt_bf16_to_fp32(ov::bfloat16* data) {
    __m256i vec_bf16 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(data));
    __m512i vec_int32 = _mm512_cvtepu16_epi32(vec_bf16);
    __m512i vec_shift = _mm512_slli_epi32(vec_int32, 16); // left shift
    __m512  vec_fp32 = _mm512_castsi512_ps(vec_shift);
    return vec_fp32;
}

__m256i cvt_fp32_to_bf16(__m512 data) {
    // cast float to integer
    __m512i vec_int32 = _mm512_castps_si512(data);

    // right shift >> 16
    __m512i vec_x = _mm512_srli_epi32(vec_int32, 16);

    // do rounding
    // LSB = x[16]
    __m512i vec_ones = _mm512_set1_epi32(0x1);
    __m512i vec_lsbs = _mm512_and_si512(vec_x, vec_ones);

    // bias is 0x7fff
    // rounding_bias = 0x7fff + LSB
    __m512i vec_bias = _mm512_set1_epi32(0x7fff);
    __m512i vec_rounding_bias = _mm512_add_epi32(vec_lsbs, vec_bias);

    // y = (vec_int32 + rounding_bias) >> 16;
    __m512i vec_y = _mm512_srli_epi32(_mm512_add_epi32(vec_rounding_bias, vec_int32), 16);

    // check NaN
    // mask
    // nan
    // Check NaN before converting back to bf16
    __mmask16 mask = _mm512_cmp_ps_mask(data, data, _CMP_ORD_Q);
    __m512i nan = _mm512_set1_epi32(0xffff);
    __m512i vec_z = _mm512_mask_blend_epi32(mask, nan, vec_y);

    // cast 32 bit to 16bit
    __m256i vec_bf16 = _mm512_cvtepi32_epi16(vec_z);
    return vec_bf16;
}
#endif

void allreduce_bfloat16(ov::bfloat16* send_buf, ov::bfloat16* recv_buf, size_t count) {
#if defined(HAVE_AVX512F)
    const size_t stride = 16;
    size_t step = count / stride;
    parallel_for(step, [&](size_t j){
      size_t i = j * stride;
      __m512 vecA, vecB, vecC;
      // bf16 to fp32
      vecA = cvt_bf16_to_fp32(send_buf + i);
      vecB = cvt_bf16_to_fp32(recv_buf + i);

      vecC = _mm512_add_ps(vecA, vecB);
      // fp32 to bf16
      __m256i bf16_v = cvt_fp32_to_bf16(vecC);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(recv_buf + i), bf16_v);
    });
    size_t tail = count & ~(stride - 1);
    for (size_t i = tail; i < count; ++i) {
      recv_buf[i] += send_buf[i];
    }
#else
    const size_t stride = 8;
    const size_t unloop = 8;
    size_t step = count / unloop;
    parallel_for(step, [&](size_t i) {
      recv_buf[i * unloop]     += send_buf[i * unloop];
      recv_buf[i * unloop + 1] += send_buf[i * unloop + 1];
      recv_buf[i * unloop + 2] += send_buf[i * unloop + 2];
      recv_buf[i * unloop + 3] += send_buf[i * unloop + 3];
      recv_buf[i * unloop + 4] += send_buf[i * unloop + 4];
      recv_buf[i * unloop + 5] += send_buf[i * unloop + 5];
      recv_buf[i * unloop + 6] += send_buf[i * unloop + 6];
      recv_buf[i * unloop + 7] += send_buf[i * unloop + 7];
    });
    size_t tail = count & ~(stride - 1);
    for (size_t i = tail; i < count; ++i) {
      recv_buf[i] += send_buf[i];
    }
#endif
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov
