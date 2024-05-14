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

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov
