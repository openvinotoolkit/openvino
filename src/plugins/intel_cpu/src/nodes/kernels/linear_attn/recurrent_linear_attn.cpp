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

template <typename T>
static void recurrent_linear_attn_paged_impl(const ov::intel_cpu::PlainTensor& query,
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
                                             float q_l2_norm_eps,
                                             float k_l2_norm_eps,
                                             bool fuse_qk_l2norm,
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
    OPENVINO_ASSERT(recurrent_state_table.m_dims[1] == v_heads && recurrent_state_table.m_dims[2] == v_head_dims &&
                        recurrent_state_table.m_dims[3] == k_head_dims,
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
        const int32_t prev_nums = (seq_interval > 0) ? (seq_past_len % seq_interval) : 0;
        OPENVINO_ASSERT(seq_blocks > 0, "[CPU] paged_gdn: each sequence must have at least one cache block");

        const int32_t block_id = block_indices.at<int32_t>({static_cast<size_t>(block_begin)});
        auto* initial_state_src = recurrent_state_table.ptr<T>(static_cast<size_t>(block_id), i_h, i_v);
        cvt_copy(init_state, initial_state_src, 1, k_head_dims, 0, 0);

        const size_t hk = i_h / group_size;

        for (int32_t token = token_begin; token < token_end; token++) {
            const auto token_u = static_cast<size_t>(token);
            for (size_t j = 0; j < k_head_dims; j++) {
                b_k[j] = static_cast<float>(key.at<T>({token_u, hk, j}));
                b_q[j] = static_cast<float>(query.at<T>({token_u, hk, j}));
            }

            if (fuse_qk_l2norm) {
                l2norm(b_k, k_head_dims, k_l2_norm_eps);
                l2norm(b_q, k_head_dims, q_l2_norm_eps);
            }

            multiply_scalar(b_q, b_q, q_scale, k_head_dims);

            float b_g = static_cast<float>(gate.at<T>({token_u, i_h}));
            float b_beta = static_cast<float>(beta.at<T>({token_u, i_h}));
            b_g = std::exp(b_g);

            multiply_scalar(init_state, init_state, b_g, k_head_dims);

            float h_k = dot_product(init_state, b_k, k_head_dims, nullptr, nullptr, nullptr, 0);

            float b_v = static_cast<float>(value.at<T>({token_u, i_h, i_v}));
            b_v = (b_v - h_k) * b_beta;

            multiply_scalar(b_k, b_k, b_v, k_head_dims);
            cvt_add(init_state, init_state, b_k, 1, k_head_dims, 0, 0, 0);

            const float b_output = dot_product(init_state, b_q, k_head_dims, nullptr, nullptr, nullptr, 0);
            output_attn.at<T>({token_u, i_h, i_v}) = static_cast<T>(b_output);

            const int32_t processed_tokens = (token - token_begin) + 1;
            const int32_t cached_tokens = prev_nums + processed_tokens;
            const bool interval_hit = (seq_interval > 0) && ((cached_tokens % seq_interval) == 0);
            const bool is_last_token = (token == token_end - 1);
            const bool should_store = interval_hit || is_last_token;
            if (should_store) {
                const int32_t slot = (seq_interval > 0) ? (1 + (cached_tokens - 1) / seq_interval) : 1;
                if (slot < seq_blocks) {
                    const int32_t block_id = block_indices.at<int32_t>({static_cast<size_t>(block_begin + slot)});
                    auto* updated_state_dst = recurrent_state_table.ptr<T>(static_cast<size_t>(block_id), i_h, i_v);
                    cvt_copy(updated_state_dst, init_state, 1, k_head_dims, 0, 0);
                }
            }
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
                                 float q_l2_norm_eps,
                                 float k_l2_norm_eps,
                                 bool fuse_qk_l2norm,
                                 ov::intel_cpu::PlainTensor& output_attn,
                                 float* temp_buffer,
                                 const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    const auto data_prc = query.get_precision();
    OPENVINO_ASSERT(key.get_precision() == data_prc && value.get_precision() == data_prc &&
                        recurrent_state_table.get_precision() == data_prc && gate.get_precision() == data_prc &&
                        beta.get_precision() == data_prc && output_attn.get_precision() == data_prc,
                    "[CPU] paged_gdn: q/k/v/state/gate/beta/output precisions must match");

    if (data_prc == ov::element::f32) {
        recurrent_linear_attn_paged_impl<float>(query,
                                                key,
                                                value,
                                                recurrent_state_table,
                                                gate,
                                                beta,
                                                subsequence_begins,
                                                block_indices,
                                                block_indices_begins,
                                                past_lens,
                                                cache_interval,
                                                q_l2_norm_eps,
                                                k_l2_norm_eps,
                                                fuse_qk_l2norm,
                                                output_attn,
                                                temp_buffer,
                                                cpu_parallel);
    } else if (data_prc == ov::element::f16) {
        recurrent_linear_attn_paged_impl<ov::float16>(query,
                                                      key,
                                                      value,
                                                      recurrent_state_table,
                                                      gate,
                                                      beta,
                                                      subsequence_begins,
                                                      block_indices,
                                                      block_indices_begins,
                                                      past_lens,
                                                      cache_interval,
                                                      q_l2_norm_eps,
                                                      k_l2_norm_eps,
                                                      fuse_qk_l2norm,
                                                      output_attn,
                                                      temp_buffer,
                                                      cpu_parallel);
    } else if (data_prc == ov::element::bf16) {
        recurrent_linear_attn_paged_impl<ov::bfloat16>(query,
                                                       key,
                                                       value,
                                                       recurrent_state_table,
                                                       gate,
                                                       beta,
                                                       subsequence_begins,
                                                       block_indices,
                                                       block_indices_begins,
                                                       past_lens,
                                                       cache_interval,
                                                       q_l2_norm_eps,
                                                       k_l2_norm_eps,
                                                       fuse_qk_l2norm,
                                                       output_attn,
                                                       temp_buffer,
                                                       cpu_parallel);
    } else {
        OPENVINO_ASSERT(false, "[CPU] paged_gdn: unsupported precision");
    }
}

}  // namespace ov::Extensions::Cpu::XARCH