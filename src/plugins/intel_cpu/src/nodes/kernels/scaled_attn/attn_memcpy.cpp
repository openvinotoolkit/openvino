// Copyright (C) 2018-2023 Intel Corporation
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
#include "common.hpp"
#include "attn_memcpy.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace XARCH {

using namespace ov;

// float16 <- float
template<typename TA, typename TB>
void attn_copy(TA* a, TB* b, size_t n) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
        auto vb = mm512_uni_loadu_ps(b + i);
        mm512_uni_storeu_ps(a + i, vb);
    }
#elif defined(HAVE_AVX2)
    for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
        auto vb = mm256_uni_loadu_ps(b + i);
        mm256_uni_storeu_ps(a + i, vb);
    }
#endif
    for (; i < n; i++) {
        a[i] = b[i];
    }
}

template <typename T, typename T2>
void attn_memcpy_kernel(const ov::intel_cpu::PlainTensor& k_input,
                        const ov::intel_cpu::PlainTensor& v_input,
                        const ov::intel_cpu::PlainTensor& past_k_output,
                        const ov::intel_cpu::PlainTensor& past_v_output) {
    size_t B = k_input.m_dims[0], H = k_input.m_dims[1], L1 = k_input.m_dims[2], S = k_input.m_dims[3];
    parallel_for3d(B, H, L1, [&](size_t b, size_t h, size_t m) {
        attn_copy(&past_k_output.at<T2>({b, h, m, 0}),
                  &k_input.at<T>({b, h, m, 0}),
                  S);
        attn_copy(&past_v_output.at<T2>({b, h, m, 0}),
                  &v_input.at<T>({b, h, m, 0}),
                  S);
    });
}

template <typename T>
void attn_memcpy_kernel(const ov::intel_cpu::PlainTensor& k_input,
                        const ov::intel_cpu::PlainTensor& v_input,
                        const ov::intel_cpu::PlainTensor& past_k_output,
                        const ov::intel_cpu::PlainTensor& past_v_output) {
    size_t B = k_input.m_dims[0], H = k_input.m_dims[1], L1 = k_input.m_dims[2], S = k_input.m_dims[3];
    parallel_for3d(B, H, L1, [&](size_t b, size_t h, size_t m) {
        memcpy(&past_k_output.at<T>({b, h, m, 0}),
               &k_input.at<T>({b, h, m, 0}),
               S * sizeof(T));
        memcpy(&past_v_output.at<T>({b, h, m, 0}),
               &v_input.at<T>({b, h, m, 0}),
               S * sizeof(T));
    });
}

void attn_memcpy(const ov::intel_cpu::PlainTensor& k_input,
                 const ov::intel_cpu::PlainTensor& v_input,
                 const ov::intel_cpu::PlainTensor& past_k_output,
                 const ov::intel_cpu::PlainTensor& past_v_output) {
    if (past_k_output.get_precision() == k_input.get_precision()) {
        if (past_k_output.get_precision() == ov::element::bf16) {
            attn_memcpy_kernel<ov::bfloat16>(k_input, v_input, past_k_output, past_v_output);
        } else {
            assert(past_k_output.get_precision() == ov::element::f16);
            attn_memcpy_kernel<ov::float16>(k_input, v_input, past_k_output, past_v_output);
        }
    } else if (past_k_output.get_precision() == ov::element::f16) {
        attn_memcpy_kernel<float, ov::float16>(k_input, v_input, past_k_output, past_v_output);
    } else {
        attn_memcpy_kernel<float, float>(k_input, v_input, past_k_output, past_v_output);
    }
}
}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine