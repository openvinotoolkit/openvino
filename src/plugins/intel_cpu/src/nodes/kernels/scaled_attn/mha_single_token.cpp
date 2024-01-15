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
#include "mha_single_token.hpp"
#include "common.hpp"
#include "softmax_kernel.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

using namespace ov;

template<typename T>
void attn_acc_value(float* out, float weight, T* v, size_t S) {
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

template<typename TA, typename TB>
float dot_product(TA* a, TB* b, size_t n) {
    size_t i = 0;
    float sum = 0.0f;
#if defined(HAVE_AVX512F)
    auto vsum = _mm512_setzero_ps();
    for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
        auto va = mm512_uni_loadu_ps(a + i);
        auto vb = mm512_uni_loadu_ps(b + i);
        vsum = _mm512_fmadd_ps(va, vb, vsum);
    }
    sum = _mm512_reduce_add_ps(vsum);
#elif defined(HAVE_AVX2)
    auto vsum = _mm256_set1_ps(0.0f);
    for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
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

template<typename T>
void attn_reduce(T* dst, float* temp, size_t M, size_t S, size_t temp_stride) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    for (; i + vec_len_f32_avx512 <= S; i+= vec_len_f32_avx512) {
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
    for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
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
    for (; i < S; i++) {
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

template <typename T, typename T2>
void mha_single_token_kernel(const ov::intel_cpu::PlainTensor& query,
                             const ov::intel_cpu::PlainTensor& present_key,
                             const ov::intel_cpu::PlainTensor& present_value,
                             const ov::intel_cpu::PlainTensor& alibi_mask,
                             const ov::intel_cpu::PlainTensor& attention_mask,
                             const ov::intel_cpu::PlainTensor& beams,
                             ov::intel_cpu::PlainTensor& output_emb,
                             ov::intel_cpu::PlainTensor& buf_attn_w,
                             ov::intel_cpu::PlainTensor& buf_attn_score,
                             bool has_out_transpose,
                             bool auto_causal,
                             float d_scale) {
    ov::intel_cpu::PlainTensor causal_mask;
    bool select_nfltmax_at_0 = false;
    auto B = query.size(0);
    auto H = query.size(1);
    auto q_len = query.size(2);
    auto S = query.size(3);
    auto kv_len = present_key.size(2);
    auto h_group_num = present_key.size(1);
    size_t h_each_group_len = 1;
    if (h_group_num != H) {
        h_each_group_len = H / h_group_num;
    }
    if (d_scale == 0.0f)
        d_scale = 1.0f / sqrt(S);

    // use per-token kernel, for each k,v token
    //  attn mask is a matrix of q_len(kv_len)
    buf_attn_w.resize<float>({B, H, q_len, kv_len});

    bool is_abcd = present_key.stride(1) >= present_key.stride(2);
    size_t dim0 = is_abcd ? B : kv_len;
    size_t dim1 = is_abcd ? h_group_num : B;
    size_t dim2 = is_abcd ? kv_len : h_group_num;

    parallel_for3d(dim0, dim1, dim2, [&](size_t d0, size_t d1, size_t d2) {
        size_t b = is_abcd ? d0 : d1;
        size_t h_group = is_abcd ? d1 : d2;
        size_t pk = is_abcd ? d2 : d0;

        // which batch item should be used at postion pk?
        auto b_kv = beams ? beams.at<int32_t>({b, pk}) : b;
        for (size_t pq = 0; pq < q_len; pq++) {
            for (size_t h = h_group * h_each_group_len; h < (h_group + 1) * h_each_group_len; h++) {
                buf_attn_w.at<float>({b, h, pq, pk}) =
                    dot_product(&query.at<T>({b, h, pq, 0}), &present_key.at<T2>({b_kv, h_group, pk, 0}, true), S);
            }
        }
    });

    parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
        // apply attention mask & sofmax
        auto ncausal = auto_causal ? (kv_len - q_len + pq + 1) : kv_len;
        float* alibi_ptr = alibi_mask ? &alibi_mask.at<float>({b, h, pq, 0}, true) : nullptr;
        float* attn_mask_ptr = attention_mask ? &attention_mask.at<float>({b, h, pq, 0}, true) : nullptr;
        uint8_t* cmask_ptr = causal_mask ? &causal_mask.at<uint8_t>({b, h, pq, 0}, true) : nullptr;
        attn_softmax_kernel(&buf_attn_w.at<float>({b, h, pq, 0}),
                            &buf_attn_w.at<float>({b, h, pq, 0}),
                            d_scale,
                            alibi_ptr,
                            attn_mask_ptr,
                            cmask_ptr,
                            select_nfltmax_at_0,
                            ncausal,
                            kv_len,
                            ov::element::f32);
    });

    // attn_w * V
    auto nthr = parallel_get_max_threads();
    buf_attn_score.resize<float>({static_cast<size_t>(nthr), B, q_len, H, S});
    // buf_attn_w {B, H, q_len, kv_len}
    parallel_nt_static(nthr, [&](const size_t ithr, const size_t nthr) {
        size_t start{0}, end{0};
        splitter(B * h_group_num * kv_len, nthr, ithr, start, end);

        memset(&buf_attn_score.at<float>({ithr, 0, 0, 0, 0}), 0, buf_attn_score.stride(0) * sizeof(float));

        size_t b, h_group, pv;
        if (start < end) {
            if (is_abcd)
                parallel_it_init(start, b, B, h_group, h_group_num, pv, kv_len);
            else
                parallel_it_init(start, pv, kv_len, b, B, h_group, h_group_num);
            for (size_t iwork = start; iwork < end; ++iwork) {
                auto b_kv = beams ? beams.at<int32_t>({b, pv}) : b;
                auto* v = &present_value.at<T2>({b_kv, h_group, pv, 0}, true);
                for (size_t pq = 0; pq < q_len; pq++) {
                    for (size_t h = h_group * h_each_group_len; h < (h_group + 1) * h_each_group_len; h++) {
                        attn_acc_value(&buf_attn_score.at<float>({ithr, b, pq, h, 0}),
                                    buf_attn_w.at<float>({b, h, pq, pv}),
                                    v,
                                    S);
                    }
                }
                if (is_abcd)
                    parallel_it_step(b, B, h_group, h_group_num, pv, kv_len);
                else
                    parallel_it_step(pv, kv_len, b, B, h_group, h_group_num);
            }
        }
    });

    parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
        auto* temp = &buf_attn_score.at<float>({0, b, pq, h, 0});
        size_t temp_stride = buf_attn_score.stride(0);
        auto* dst = has_out_transpose ? &output_emb.at<T>({b, pq, h * S}) : &output_emb.at<T>({b, h, pq});
        attn_reduce(dst, temp, nthr, S, temp_stride);
    });
}

void mha_single_token(const ov::intel_cpu::PlainTensor& query,
                      const ov::intel_cpu::PlainTensor& present_key,
                      const ov::intel_cpu::PlainTensor& present_value,
                      const ov::intel_cpu::PlainTensor& alibi_mask,
                      const ov::intel_cpu::PlainTensor& attention_mask,
                      const ov::intel_cpu::PlainTensor& beams,
                      ov::intel_cpu::PlainTensor& output_emb,
                      ov::intel_cpu::PlainTensor& buf_attn_w,
                      ov::intel_cpu::PlainTensor& buf_attn_score,
                      bool has_out_transpose,
                      bool auto_causal,
                      float d_scale) {
    if (query.get_precision() == ov::element::bf16) {
        mha_single_token_kernel<ov::bfloat16, ov::bfloat16>(query,
                                                            present_key,
                                                            present_value,
                                                            alibi_mask,
                                                            attention_mask,
                                                            beams,
                                                            output_emb,
                                                            buf_attn_w,
                                                            buf_attn_score,
                                                            has_out_transpose,
                                                            auto_causal,
                                                            d_scale);
    } else if (query.get_precision() == ov::element::f32) {
        if (present_key.get_precision() == ov::element::f16) {
            mha_single_token_kernel<float, ov::float16>(query,
                                                        present_key,
                                                        present_value,
                                                        alibi_mask,
                                                        attention_mask,
                                                        beams,
                                                        output_emb,
                                                        buf_attn_w,
                                                        buf_attn_score,
                                                        has_out_transpose,
                                                        auto_causal,
                                                        d_scale);
        } else {
            mha_single_token_kernel<float, float>(query,
                                                present_key,
                                                present_value,
                                                alibi_mask,
                                                attention_mask,
                                                beams,
                                                output_emb,
                                                buf_attn_w,
                                                buf_attn_score,
                                                has_out_transpose,
                                                auto_causal,
                                                d_scale);
        }
    } else {
        OPENVINO_THROW("Unsupported precision: ", query.get_precision());
    }
}
}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov