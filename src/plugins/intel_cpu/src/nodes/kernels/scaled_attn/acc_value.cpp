// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <assert.h>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include "openvino/core/type/bfloat16.hpp"
#include "common.hpp"
#include "acc_value.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace XARCH {

template<typename T>
void attn_acc_value_inner(float* out, float weight, T* v, size_t S) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    auto attn_w_vec_fp32 = _mm512_set1_ps(weight);
    for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
        auto v_value = mm512_uni_loadu_ps(v + i);
        auto v_out = mm512_uni_loadu_ps(out + i);
        v_out = _mm512_fmadd_ps(attn_w_vec_fp32, v_value, v_out);
        _mm512_storeu_ps(out + i, v_out);
    }
#elif defined(HAVE_AVX2)
    auto attn_w_vec_fp32 = _mm256_set1_ps(weight);
    for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
        auto v_value = mm256_uni_loadu_ps(v + i);
        auto v_out = mm256_uni_loadu_ps(out + i);
        v_out = _mm256_fmadd_ps(attn_w_vec_fp32, v_value, v_out);
        mm256_uni_storeu_ps(out + i, v_out);
    }
#endif
    for (; i < S; i++) {
        out[i] += weight * v[i];
    }
}

void attn_acc_values(float** outs, float* weights, void** vs, size_t vec_num, size_t vec_len, ov::element::Type input_precision) {
    if (input_precision == ov::element::f32) {
        for (size_t i = 0; i < vec_num; i++) {
            auto out_ptr = outs[i];
            auto v_ptr = static_cast<float*>(vs[i]);
            attn_acc_value_inner(out_ptr, weights[i], v_ptr, vec_len);
        }
    } else {
        assert(input_precision == ov::element::bf16);
        for (size_t i = 0; i < vec_num; i++) {
            auto out_ptr = outs[i];
            auto v_ptr = static_cast<ov::bfloat16*>(vs[i]);
            attn_acc_value_inner(out_ptr, weights[i], v_ptr, vec_len);
        }
    }
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine