// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif
#include "nodes/kernels/scaled_attn/common.hpp"
#include "recurrent_linear_attn.hpp"

namespace ov::Extensions::Cpu::XARCH {

static inline void l2norm(float* a, size_t n, float eps) {
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
    float inv = 1.0f / std::sqrt(sum + eps);
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
    hsum(vsum);
    sum = _mm256_cvtss_f32(vsum);
    for (; i < n; ++i) {
        sum += a[i] * a[i];
    }
    float inv = 1.0f / std::sqrt(sum + eps);
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
    float inv = 1.0f / std::sqrt(sum + eps);
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
                           float q_l2_norm_eps,
                           float k_l2_norm_eps,
                           bool fuse_qk_l2norm,
                           ov::intel_cpu::PlainTensor& output_attn,
                           ov::intel_cpu::PlainTensor& output_recurrent_state,
                           float* temp_buffer,
                           const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    size_t B = query.m_dims[0];
    size_t T = query.m_dims[1];
    size_t H = query.m_dims[2];
    size_t K = query.m_dims[3];
    size_t V = value.m_dims[3];
    const size_t K_HEAD_DIMS = K;
    const size_t V_HEAD_DIMS = V;
    const float q_scale = 1 / std::sqrt(static_cast<float>(K_HEAD_DIMS));
    cpu_parallel->parallel_for3d(B, H, V, [&](size_t i_b, size_t i_h, size_t i_v) {
        size_t tid = parallel_get_thread_num();
        float* init_state = temp_buffer + tid * 3 * K_HEAD_DIMS;
        float* b_k = temp_buffer + tid * 3 * K_HEAD_DIMS + K_HEAD_DIMS;
        float* b_q = temp_buffer + tid * 3 * K_HEAD_DIMS + 2 * K_HEAD_DIMS;
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
            for (size_t j = 0; j < K_HEAD_DIMS; j++) {
                b_k[j] = k_ptr[i * H * K_HEAD_DIMS + j];
                b_q[j] = q_ptr[i * H * K_HEAD_DIMS + j];
            }
            if (fuse_qk_l2norm) {
                l2norm(b_k, K_HEAD_DIMS, k_l2_norm_eps);
                l2norm(b_q, K_HEAD_DIMS, q_l2_norm_eps);
            }
            multiply_scalar(b_q, b_q, q_scale, K_HEAD_DIMS);
            // h0 * gate
            multiply_scalar(init_state, init_state, b_g, K_HEAD_DIMS);
            float h_k = dot_product(init_state, b_k, K_HEAD_DIMS, nullptr, nullptr, nullptr, 0);
            // B, T, H, V
            float b_v = v_ptr[i_v + i * H * V_HEAD_DIMS];
            b_v -= h_k;
            // b_v * b_k
            b_v *= b_beta;
            multiply_scalar(b_k, b_k, b_v, K_HEAD_DIMS);
            // h = h0 + update
            cvt_add(init_state, init_state, b_k, 1, K_HEAD_DIMS, 0, 0, 0);
            float b_output = dot_product(init_state, b_q, K_HEAD_DIMS, nullptr, nullptr, nullptr, 0);
            output_attn.at<float>({i_b, i, i_h, i_v}) = b_output;
        }
        for (size_t j = 0; j < K_HEAD_DIMS; j++) {
            output_recurrent_state.at<float>({i_b, i_h, j, i_v}) = init_state[j];
        }
    });
}

// TODO merge 2 functions since original gdn is a special case of paged gdn
void recurrent_linear_attn_paged(const ov::intel_cpu::PlainTensor& query,
                                 const ov::intel_cpu::PlainTensor& key,
                                 const ov::intel_cpu::PlainTensor& value,
                                 ov::intel_cpu::PlainTensor& recurrent_state_table,
                                 const ov::intel_cpu::PlainTensor& gate,
                                 const ov::intel_cpu::PlainTensor& beta,
                                 const ov::intel_cpu::PlainTensor& subsequence_begins,
                                 const ov::intel_cpu::PlainTensor& block_indices,
                                 const ov::intel_cpu::PlainTensor& block_indices_begins,
                                 const ov::intel_cpu::PlainTensor& past_lens,
                                 const ov::intel_cpu::PlainTensor& cache_interval,
                                 ov::intel_cpu::PlainTensor& output_attn,
                                 float* temp_buffer,
                                 const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    const size_t tokens = query.m_dims[0];
    const size_t qk_heads = query.m_dims[1];
    const size_t k_head_dims = query.m_dims[2];
    const size_t v_heads = value.m_dims[1];
    const size_t v_head_dims = value.m_dims[2];

    OPENVINO_ASSERT(key.m_dims[0] == tokens && key.m_dims[1] == qk_heads && key.m_dims[2] == k_head_dims,
                    "[CPU] paged_gdn: key shape mismatch with query");
    OPENVINO_ASSERT(value.m_dims[0] == tokens && value.m_dims[1] == v_heads,
                    "[CPU] paged_gdn: value shape mismatch with query tokens/heads");
    OPENVINO_ASSERT(recurrent_state_table.m_dims[1] == v_heads && recurrent_state_table.m_dims[2] == k_head_dims &&
                        recurrent_state_table.m_dims[3] == v_head_dims,
                    "[CPU] paged_gdn: recurrent_state_table shape mismatch");

    const auto seq_tensor_size = subsequence_begins.m_dims[0];
    OPENVINO_ASSERT(seq_tensor_size >= 1, "[CPU] paged_gdn: subsequence_begins must be non-empty");
    const size_t num_sequences = seq_tensor_size - 1;
    const size_t group_size = v_heads / qk_heads;
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(k_head_dims));

    cpu_parallel->parallel_for3d(num_sequences, v_heads, v_head_dims, [&](size_t seq, size_t i_h, size_t i_v) {
        size_t tid = parallel_get_thread_num();
        float* init_state = temp_buffer + tid * 3 * k_head_dims;
        float* b_k = temp_buffer + tid * 3 * k_head_dims + k_head_dims;
        float* b_q = temp_buffer + tid * 3 * k_head_dims + 2 * k_head_dims;

        const int32_t token_begin = subsequence_begins.at<int32_t>({seq});
        const int32_t token_end = subsequence_begins.at<int32_t>({seq + 1});
        const int32_t block_begin = block_indices_begins.at<int32_t>({seq});
        const int32_t block_end = block_indices_begins.at<int32_t>({seq + 1});
        const int32_t seq_blocks = std::max(block_end - block_begin, 0);
        const int32_t seq_past_len = past_lens.at<int32_t>({seq});
        const int32_t seq_interval = cache_interval.at<int32_t>({seq});

        for (size_t j = 0; j < k_head_dims; j++) {
            init_state[j] = 0.0f;
        }

        if (seq_interval > 0 && seq_blocks > 0 && seq_past_len > 0) {
            const int32_t read_slot = 0;
            const int32_t block_id = block_indices.at<int32_t>({static_cast<size_t>(block_begin + read_slot)});
            for (size_t j = 0; j < k_head_dims; j++) {
                init_state[j] = recurrent_state_table.at<float>({static_cast<size_t>(block_id), i_h, j, i_v});
            }
        }

        const size_t hk = i_h / group_size;

        for (int32_t token = token_begin; token < token_end; token++) {
            const auto token_u = static_cast<size_t>(token);
            for (size_t j = 0; j < k_head_dims; j++) {
                b_k[j] = key.at<float>({token_u, hk, j});
                b_q[j] = query.at<float>({token_u, hk, j});
            }

            l2norm(b_k, k_head_dims, 1e-6f);
            l2norm(b_q, k_head_dims, 1e-6f);

            multiply_scalar(b_q, b_q, q_scale, k_head_dims);

            float b_g = gate.at<float>({token_u, i_h});
            float b_beta = beta.at<float>({token_u, i_h});
            b_g = std::exp(b_g);

            multiply_scalar(init_state, init_state, b_g, k_head_dims);

            float h_k = dot_product(init_state, b_k, k_head_dims, nullptr, nullptr, nullptr, 0);

            float b_v = value.at<float>({token_u, i_h, i_v});
            b_v = (b_v - h_k) * b_beta;

            multiply_scalar(b_k, b_k, b_v, k_head_dims);
            cvt_add(init_state, init_state, b_k, 1, k_head_dims, 0, 0, 0);

            const float b_output = dot_product(init_state, b_q, k_head_dims, nullptr, nullptr, nullptr, 0);
            output_attn.at<float>({token_u, i_h, i_v}) = b_output;

            if (seq_interval > 0 && seq_blocks > 0) {
                const int32_t local_token_idx = token - token_begin;
                const int32_t processed_tokens = local_token_idx + 1;
                const bool should_store = ((processed_tokens % seq_interval) == 0) || (token == token_end - 1);
                if (should_store) {
                    const int32_t slot = (processed_tokens + seq_interval - 1) / seq_interval;
                    if (slot < seq_blocks) {
                        const int32_t block_id = block_indices.at<int32_t>({static_cast<size_t>(block_begin + slot)});
                        for (size_t j = 0; j < k_head_dims; j++) {
                            recurrent_state_table.at<float>({static_cast<size_t>(block_id), i_h, j, i_v}) =
                                init_state[j];
                        }
                    }
                }
            }
        }
    });
}

}  // namespace ov::Extensions::Cpu::XARCH