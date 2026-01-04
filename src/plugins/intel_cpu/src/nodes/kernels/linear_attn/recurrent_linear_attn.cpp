// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"


#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif
#include "nodes/kernels/scaled_attn/softmax_kernel.hpp"
#include "recurrent_linear_attn.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"

namespace ov::Extensions::Cpu::XARCH {

float dot_product(float* a, float* b, size_t n) {
float result = 0.0f;
size_t i = 0;
#if defined(HAVE_AVX512F)
    auto vsum0 = _mm512_setzero_ps();
    auto vsum1 = _mm512_setzero_ps();
    auto vsum2 = _mm512_setzero_ps();
    auto vsum3 = _mm512_setzero_ps();
    for (; i + 4 * vec_len_f32_avx512 <= n; i += 4 * vec_len_f32_avx512) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);
        auto va2 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 2);
        auto va3 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 3);

        auto vb0 = mm512_uni_loadu_ps(b + i);
        auto vb1 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512);
        auto vb2 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512 * 2);
        auto vb3 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512 * 3);

        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
        vsum2 = _mm512_fmadd_ps(va2, vb2, vsum2);
        vsum3 = _mm512_fmadd_ps(va3, vb3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);

        auto vb0 = mm512_uni_loadu_ps(b + i);
        auto vb1 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512);

        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
        i += 2 * vec_len_f32_avx512;
    }
    if (i + vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto vb0 = mm512_uni_loadu_ps(b + i);
        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        i += vec_len_f32_avx512;
    }
    vsum0 = _mm512_add_ps(vsum0, vsum1);
    vsum2 = _mm512_add_ps(vsum2, vsum3);
    vsum0 = _mm512_add_ps(vsum0, vsum2);
    result = _mm512_reduce_add_ps(vsum0);
#else
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
#endif
    return result;
}

void scale(float *a, float scale, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] = a[i] * scale;
    }
}

void add(float *a, float* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] += b[i];
    }
}

void recurrent_linear_attn(const ov::intel_cpu::PlainTensor& query,
                      const ov::intel_cpu::PlainTensor& key,
                      const ov::intel_cpu::PlainTensor& value,
                      const ov::intel_cpu::PlainTensor& beta,
                      const ov::intel_cpu::PlainTensor& g,
                      const ov::intel_cpu::PlainTensor& initial_states,
                      ov::intel_cpu::PlainTensor& output,
                      ov::intel_cpu::PlainTensor& output_hidden_states) {
	size_t B = query.m_dims[0];
	size_t T = query.m_dims[1];
	size_t H = query.m_dims[2];
	size_t K = query.m_dims[3];
	size_t V = value.m_dims[3];
	// printf(" B %ld T %ld H %ld K %ld V %ld \n", B, T, H, K, V);
	constexpr size_t K_HEAD_DIMS = 128;
	constexpr size_t V_HEAD_DIMS = 128;
	ov::parallel_for3d(B, H, V, [&](size_t i_b, size_t i_h, size_t i_v) {
		float init_state[128] = {0};
		float b_k[128] = {0};
		float b_q[128] = {0};
		// B, T, H, K
		float* q_ptr = query.ptr<float>(i_b, 0, i_h);
		float* k_ptr = key.ptr<float>(i_b, 0, i_h);
		float* v_ptr = value.ptr<float>(i_b, 0, i_h);
		// B, H, K, V
		for (size_t j = 0; j < K_HEAD_DIMS; j++) {
			init_state[j] = initial_states.at<float>({i_b, i_h, j, i_v});
		}
		for (size_t i = 0; i < T; i++) {
			// g: B, T, H
			float b_g = g.at<float>({i_b, i, i_h});
			float b_beta = beta.at<float>({i_b, i, i_h});
			b_g = exp(b_g);
			for (int j = 0; j < K_HEAD_DIMS; j++) {
				b_k[j] = k_ptr[i * K_HEAD_DIMS  + j];
				b_q[j] = q_ptr[i * K_HEAD_DIMS  + j] * 1 / sqrt(128);
			}
			// h0 * g
			// scale(init_state, b_g, K_HEAD_DIMS);
            multiply_scalar(init_state, init_state, b_g, K_HEAD_DIMS);
			float h_k = dot_product(init_state, b_k, K_HEAD_DIMS);
			float b_v = v_ptr[i_v + i * K_HEAD_DIMS];
			b_v -= h_k;
			// b_v * b_k
			b_v *= b_beta;
			// scale(b_k, b_v, K_HEAD_DIMS);
            multiply_scalar(b_k, b_k, b_v, K_HEAD_DIMS);
			// h = h0 + update
			add(init_state, b_k, K_HEAD_DIMS);
			float b_output  = dot_product(init_state, b_q, K_HEAD_DIMS);
			output.at<float>({i_b, i, i_h, i_v}) = b_output;
		}
		for (size_t j = 0; j < K_HEAD_DIMS; j++) {
			initial_states.at<float>({i_b, i_h, j, i_v}) = init_state[j];
		}
	});
}

} // namespace ov::Extensions::Cpu::XARCH