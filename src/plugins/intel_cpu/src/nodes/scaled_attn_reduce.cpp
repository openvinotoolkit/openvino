// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <float.h>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
    #include <x86intrin.h>
    #include <immintrin.h>
#endif

#include "openvino/core/type/bfloat16.hpp"
#include "scaled_attn_common.hpp"
#include "scaled_attn_reduce.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace XARCH {

template<typename T>
void attn_reduce_inner(T* dst, float* temp, size_t M, size_t S, size_t temp_stride) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    for (; i <= S - 16; i+= 16) {
        auto* src = temp + i;
        auto result_vec_fp32 = _mm512_setzero_ps();
        for (size_t m = 0; m < M; m++) {
            //auto* temp = &m_temp.at({ithr, b, pq, h, 0});
            auto o_vec_fp32 = _mm512_loadu_ps(src);
            result_vec_fp32 = _mm512_add_ps(result_vec_fp32, o_vec_fp32);
            src += temp_stride;
        }
        // save to bf16
        mm512_uni_storeu_ps(dst + i, result_vec_fp32);
    }
#elif defined(HAVE_AVX2)
    for (; i <= S - 8; i += 8) {
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
    for (; i <S; i++) {
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

void attn_reduce(void* dst, float* temp, size_t M, size_t S, size_t temp_stride, Precision input_precision) {
    if (input_precision == Precision::FP32) {
        auto dst_ptr = static_cast<float*>(dst);
        attn_reduce_inner(dst_ptr, temp, M, S, temp_stride);
    } else if (input_precision == Precision::BF16) {
        auto dst_ptr = static_cast<ov::bfloat16*>(dst);
        attn_reduce_inner(dst_ptr, temp, M, S, temp_stride);
    }
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine