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

inline float dot_product(float* a, float* b, size_t n) {
#if defined(HAVE_AVX512F)
	size_t i = 0;
	__m512 vsum = _mm512_setzero_ps();
	for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
		__m512 va = _mm512_loadu_ps(a + i);
		__m512 vb = _mm512_loadu_ps(b + i);
		vsum = _mm512_fmadd_ps(va, vb, vsum);
	}
	float result = _mm512_reduce_add_ps(vsum);
	for (; i < n; ++i) {
		result += a[i] * b[i];
	}
	return result;
#elif defined(HAVE_AVX2)
	size_t i = 0;
	__m256 vsum = _mm256_setzero_ps();
	for (; i + 8 <= n; i += 8) {
		__m256 va = _mm256_loadu_ps(a + i);
		__m256 vb = _mm256_loadu_ps(b + i);
		vsum = _mm256_fmadd_ps(va, vb, vsum);
	}
	float result = hsum(vsum);
	for (; i < n; ++i) {
		result += a[i] * b[i];
	}
	return result;
#else
	float result = 0.0f;
	for (size_t i = 0; i < n; ++i) {
		result += a[i] * b[i];
	}
	return result;
#endif
}

inline void add(float* a, float* b, size_t n) {
#if defined(HAVE_AVX512F)
	size_t i = 0;
	for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
		__m512 va = _mm512_loadu_ps(a + i);
		__m512 vb = _mm512_loadu_ps(b + i);
		va = _mm512_add_ps(va, vb);
		_mm512_storeu_ps(a + i, va);
	}
	for (; i < n; ++i) {
		a[i] += b[i];
	}
#elif defined(HAVE_AVX2)
	size_t i = 0;
	for (; i + 8 <= n; i += 8) {
		__m256 va = _mm256_loadu_ps(a + i);
		__m256 vb = _mm256_loadu_ps(b + i);
		va = _mm256_add_ps(va, vb);
		_mm256_storeu_ps(a + i, va);
	}
	for (; i < n; ++i) {
		a[i] += b[i];
	}
#else
	for (size_t i = 0; i < n; i++) {
		a[i] += b[i];
	}
#endif
}

static inline void l2norm(float* a, size_t n) {
	float sum = 0.0f;
#if defined(HAVE_AVX512F)
	size_t i = 0;
	__m512 vsum = _mm512_setzero_ps();
	for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
		__m512 v = _mm512_loadu_ps(a + i);
		vsum = _mm512_fmadd_ps(v, v, vsum);
	}
	sum = _mm512_reduce_add_ps(vsum);
	for (; i < n; ++i) {
		sum += a[i] * a[i];
	}
	float inv = 1.0f / std::sqrt(sum + 1e-6f);
	__m512 vscale = _mm512_set1_ps(inv);
	i = 0;
	for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
		__m512 v = _mm512_loadu_ps(a + i);
		v = _mm512_mul_ps(v, vscale);
		_mm512_storeu_ps(a + i, v);
	}
	for (; i < n; ++i) {
		a[i] *= inv;
	}
#elif defined(HAVE_AVX2)
	size_t i = 0;
	__m256 vsum = _mm256_setzero_ps();
	for (; i + 8 <= n; i += 8) {
		__m256 v = _mm256_loadu_ps(a + i);
		vsum = _mm256_fmadd_ps(v, v, vsum);
	}
    float sum = hsum(vsum);
	for (; i < n; ++i) {
		sum += a[i] * a[i];
	}
	float inv = 1.0f / std::sqrt(sum + 1e-6f);
	__m256 vscale = _mm256_set1_ps(inv);
	i = 0;
	for (; i + 8 <= n; i += 8) {
		__m256 v = _mm256_loadu_ps(a + i);
		v = _mm256_mul_ps(v, vscale);
		_mm256_storeu_ps(a + i, v);
	}
	for (; i < n; ++i) {
		a[i] *= inv;
	}
#else
	for (size_t j = 0; j < n; j++) {
		sum += a[j] * a[j];
	}
	float inv = 1.0f / std::sqrt(sum + 1e-6f);
	for (size_t j = 0; j < n; j++) {
		a[j] *= inv;
	}
#endif
}

void recurrent_linear_attn(const ov::intel_cpu::PlainTensor& query,
                      const ov::intel_cpu::PlainTensor& key,
                      const ov::intel_cpu::PlainTensor& value,
                      const ov::intel_cpu::PlainTensor& recurrent_state,
                      const ov::intel_cpu::PlainTensor& gate,
                      const ov::intel_cpu::PlainTensor& beta,
                      ov::intel_cpu::PlainTensor& output_attn,
                      ov::intel_cpu::PlainTensor& output_recurrent_state) {
	size_t B = query.m_dims[0];
	size_t T = query.m_dims[1];
	size_t H = query.m_dims[2];
	size_t K = query.m_dims[3];
	size_t V = value.m_dims[3];
	const size_t K_HEAD_DIMS = K;
	const size_t V_HEAD_DIMS = V;
    const float q_scale = 1 / std::sqrt(static_cast<float>(K_HEAD_DIMS));
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
			init_state[j] = recurrent_state.at<float>({i_b, i_h, j, i_v});
		}

		for (size_t i = 0; i < T; i++) {
			// gate: B, T, H
			float b_g = gate.at<float>({i_b, i, i_h});
			float b_beta = beta.at<float>({i_b, i, i_h});
			b_g = exp(b_g);
			for (int j = 0; j < K_HEAD_DIMS; j++) {
				b_k[j] = k_ptr[i * H * K_HEAD_DIMS  + j];
				b_q[j] = q_ptr[i * H * K_HEAD_DIMS  + j];
			}
            l2norm(b_k, K_HEAD_DIMS);
            l2norm(b_q, K_HEAD_DIMS);
            multiply_scalar(b_q, b_q, q_scale, K_HEAD_DIMS);
			// h0 * gate
            // scale(init_state, b_g, K_HEAD_DIMS);
            multiply_scalar(init_state, init_state, b_g, K_HEAD_DIMS);
			float h_k = dot_product(init_state, b_k, K_HEAD_DIMS);
            // B, T, H, V
			float b_v = v_ptr[i_v + i * H * K_HEAD_DIMS];
			b_v -= h_k;
			// b_v * b_k
			b_v *= b_beta;
			// scale(b_k, b_v, K_HEAD_DIMS);
            multiply_scalar(b_k, b_k, b_v, K_HEAD_DIMS);
			// h = h0 + update
			add(init_state, b_k, K_HEAD_DIMS);
			float b_output  = dot_product(init_state, b_q, K_HEAD_DIMS);
			output_attn.at<float>({i_b, i, i_h, i_v}) = b_output;
		}
		for (size_t j = 0; j < K_HEAD_DIMS; j++) {
			output_recurrent_state.at<float>({i_b, i_h, j, i_v}) = init_state[j];
		}
	});
}

} // namespace ov::Extensions::Cpu::XARCH