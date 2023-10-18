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
#    include <immintrin.h>
#endif

#include "openvino/core/type/bfloat16.hpp"
#include "scaled_attn_common.hpp"
#include "scaled_attn_dot_product.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace XARCH {

template<typename T>
float dot_product_inner(T* a, T* b, size_t n) {
    size_t i = 0;
    float sum = 0.0f;
#if defined(HAVE_AVX512F)
    auto vsum = _mm512_setzero_ps();
    for (; i <= n - 16; i += 16) {
        auto va = mm512_uni_loadu_ps(a + i);
        auto vb = mm512_uni_loadu_ps(b + i);
        vsum = _mm512_fmadd_ps(va, vb, vsum);
    }
    sum = _mm512_reduce_add_ps(vsum);
#elif defined(HAVE_AVX2)
    auto vsum = _mm256_set1_ps(0.0f);
    for (; i <= n - 8; i += 8) {
        auto va = mm256_uni_loadu_ps(a + i);
        auto vb = mm256_uni_loadu_ps(b + i);
        vsum = _mm256_fmadd_ps(va, vb, vsum);
    }
    hsum(vsum);
    sum = _mm256_cvtss_f32(vsum);
#endif
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

float attn_dot_product(void* a, void* b, size_t len, Precision input_precision) {
    if (input_precision == Precision::FP32) {
        auto a_ptr = static_cast<float*>(a);
        auto b_ptr = static_cast<float*>(b);
        return dot_product_inner(a_ptr, b_ptr, len);
    } else if (input_precision == Precision::BF16) {
        auto a_ptr = static_cast<ov::bfloat16*>(a);
        auto b_ptr = static_cast<ov::bfloat16*>(b);
        return dot_product_inner(a_ptr, b_ptr, len);
    }
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine