// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "nodes/kernels/scaled_attn/common.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "paged_attn_kernel.hpp"
#include "softmax_kernel.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"
#include "nodes/rope.h"
#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#if defined(OPENVINO_ARCH_X86_64)
#    include "nodes/kernels/x64/brgemm_kernel.hpp"
#elif defined(OPENVINO_ARCH_ARM64) && defined(HAVE_SVE)
#    include "arm_sve.h"
#    include "nodes/kernels/aarch64/brgemm_kernel.hpp"
#    include "nodes/kernels/aarch64/sve_utils.hpp"
#    include "nodes/kernels/kai/kleidi_kernel.hpp"
#endif

#include <cstddef>
#include <cstdint>

namespace ov::Extensions::Cpu::XARCH {

#if defined(HAVE_AVX2)
inline int32_t hsum_epi32_avx(__m128i x) {
    __m128i hi64 =
        _mm_unpackhi_epi64(x, x);  // 3-operand non-destructive AVX lets us save a byte without needing a movdqa
    __m128i sum64 = _mm_add_epi32(hi64, x);
    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));  // Swap the low two elements
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);  // movd
}

// only needs AVX2
inline int32_t hsum_8x32(__m256i v) {
    __m128i sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(v),
        _mm256_extracti128_si256(v, 1));  // silly GCC uses a longer AXV512VL instruction if AVX512 is enabled :/
    return hsum_epi32_avx(sum128);
}
#endif
#if defined(HAVE_AVX512F)
// AVX512
inline int32_t hsum_16x32(__m512i v) {
    __m256i sum256 =
        _mm256_add_epi32(_mm512_castsi512_si256(v),         // low half
                         _mm512_extracti64x4_epi64(v, 1));  // high half.  AVX512F.  32x8 version is AVX512DQ
    return hsum_8x32(sum256);
}
#endif

inline void
dot_product_block_s8s8_f32(int8_t* a, int8_t* b, float* c, int32_t* c_i32, float c_scale, size_t n, size_t block_size) {
    const size_t b_stride = n + sizeof(float);
    for (size_t j = 0; j < block_size; j++) {
        const size_t params_offset = sizeof(float);
        float* scale_a = reinterpret_cast<float*>(a);
        float* scale_b = reinterpret_cast<float*>(b);
        int8_t* a_src = a + params_offset;
        int8_t* b_src = b + params_offset;
        int32_t sum = 0;
        size_t i = 0;
#if defined(HAVE_AVX512F)
        __m512i sum_v512 = _mm512_setzero_epi32();
        for (; i + vec_len_epi8_avx2 <= n; i += vec_len_epi8_avx2) {
            auto a_0 = _mm256_loadu_si256((__m256i*)(a_src + i));
            auto b_0 = _mm256_loadu_si256((__m256i*)(b_src + i));
            auto a_0_i = _mm512_cvtepi8_epi16(a_0);
            auto b_0_i = _mm512_cvtepi8_epi16(b_0);
            auto temp = _mm512_madd_epi16(a_0_i, b_0_i);
            sum_v512 = _mm512_add_epi32(sum_v512, temp);
        }
        sum += _mm512_reduce_add_epi32(sum_v512);
#elif defined(HAVE_AVX2)
        __m256i sum_v256 = _mm256_setzero_si256();
        for (; i + vec_len_epi8_avx2 / 2 <= n; i += vec_len_epi8_avx2 / 2) {
            auto a_0 = _mm_loadu_si128((__m128i*)(a_src + i));
            auto b_0 = _mm_loadu_si128((__m128i*)(b_src + i));
            auto a_0_i = _mm256_cvtepi8_epi16(a_0);
            auto b_0_i = _mm256_cvtepi8_epi16(b_0);
            auto temp = _mm256_madd_epi16(a_0_i, b_0_i);
            sum_v256 = _mm256_add_epi32(sum_v256, temp);
        }
        sum += hsum_8x32(sum_v256);
#endif
        for (; i < n; i++) {
            sum += a_src[i] * b_src[i];
        }
        *c_i32++ = sum;
        float f32_sum = static_cast<float>(sum) * scale_a[0] * scale_b[0];
        b += b_stride;
        *c++ = f32_sum * c_scale;
    }
}

void sage_attn_transpose_k(const ReorderWorkItem& item,
                           const size_t hk,
                           const size_t block_size,
                           const std::shared_ptr<ov::intel_cpu::BrgemmKernel>& brgemm_ptr,
                           ov::intel_cpu::PlainTensor& key_cache,
                           ov::intel_cpu::PlainTensor& qk_scratch_b) {
    const auto batch_in_seq = item.batch_in_seq;
    const auto batch_in_reorder = item.batch_in_reorder;
    const auto kv_block = item.kv_block_id;
    const auto block_number = item.block_number;
    const auto S = key_cache.m_dims[3] - sizeof(float);
    if (block_number < 0) {
        return;
    }
    const size_t valid_len = item.valid_block_len;
    // indexing as i8
    auto* k_ptr = key_cache.ptr<int8_t, ov::element::i8>(block_number, hk, 0, sizeof(float));
    for (size_t i = valid_len; i < block_size; i++) {
        memset(key_cache.ptr<int8_t, ov::element::i8>(block_number, hk, i), 0, sizeof(int8_t) * key_cache.m_dims[3]);
    }
    auto* repacked = qk_scratch_b.ptr<int8_t>(batch_in_reorder, kv_block, hk);
    brgemm_ptr->copy_buffer_b(k_ptr, repacked);
    // layout of repacked_data
    // block_size * S int8(quantized key)
    // block_size * scales (FP32)
    //copy b_scale to dst tensor
    float* scales =  reinterpret_cast<float*>(qk_scratch_b.ptr<int8_t>(batch_in_reorder, kv_block, hk, block_size * S));
    for(size_t i = 0; i < valid_len; i++) {
        scales[i] = reinterpret_cast<float*>(key_cache.ptr<int8_t, ov::element::i8>(block_number, hk, i, 0))[0];
    }
}

template <typename DATA_TYPE, ov::element::Type_t KEY_PREC>
void sage_attn_quantize_q(const ov::intel_cpu::PlainTensor& q,
                   ov::intel_cpu::PlainTensor& quantized_q,
                   const ov::intel_cpu::PlainTensor& past_lens,
                   const ov::intel_cpu::PlainTensor& subsequence_begins) {
    size_t H = q.m_dims[1];
    size_t S = q.m_dims[3];
    parallel_for2d(past_lens.size(0), H, [&](size_t sub_seq_id, size_t h) {
        const auto q_len =
            subsequence_begins.ptr<int32_t>()[sub_seq_id + 1] - subsequence_begins.ptr<int32_t>()[sub_seq_id];
        const auto batch_in_token = subsequence_begins.ptr<int32_t>()[sub_seq_id];
        if (q_len > 1) {
            parallel_for(q_len, [&](int32_t l) {
                quantize_q_by_dims<DATA_TYPE, KEY_PREC>(q, quantized_q, batch_in_token + l, h, S);
            });
        }
    });
}

// template <typename DATA_TYPE, ov::element::Type_t KEY_PREC, ov::element::Type_t VALUE_PREC>
// void sage_attn_ref(const ov::intel_cpu::PlainTensor& q,
//                    ov::intel_cpu::PlainTensor& k_cache,
//                    const ov::intel_cpu::PlainTensor& v_cache,
//                    const ov::intel_cpu::PlainTensor& output_emb,
//                    const ov::intel_cpu::PlainTensor& output_score,
//                    [[maybe_unused]] size_t max_context_len,
//                    const ov::intel_cpu::PlainTensor& past_lens,
//                    const ov::intel_cpu::PlainTensor& subsequence_begins,
//                    const ov::intel_cpu::PlainTensor& block_indices,
//                    const ov::intel_cpu::PlainTensor& block_indices_begins,
//                    const ov::intel_cpu::PlainTensor& alibi_slopes,
//                    const WorkItems work_items,
//                    const std::vector<std::shared_ptr<ov::intel_cpu::BrgemmKernel>>& qk_s8s8_gemm,
//                    ov::intel_cpu::PlainTensor& qk_scratch_b,
//                    std::vector<size_t>& wsp,
//                    ov::intel_cpu::PlainTensor& temp_weight_blocked,
//                    ov::intel_cpu::PlainTensor& temp_weight,
//                    ov::intel_cpu::PlainTensor& temp_output) {
//     // printf("Going to Do SageAttn\n");
//     auto seq_count = static_cast<int32_t>(past_lens.m_dims[0]);
//     int32_t block_size = 32;
//     size_t B = q.m_dims[0], H = q.m_dims[1], S = q.m_dims[2], SV = v_cache.m_dims[3] - sizeof(float) * 2;
//     size_t Hk = k_cache.m_dims[1];
//     size_t h_each_group_len = H / Hk;
//     const size_t param_size = sizeof(float);
//     const float _d_scale = 1 / sqrt(S - param_size);
//     temp_output.resize<float>({B, H, SV});
//     memset(temp_output.ptr<float>(), 0, B * H * SV * sizeof(float));
//     temp_weight.resize<float>({H, B, ov::intel_cpu::rnd_up(max_context_len, std::max(block_size, int32_t{16}))});
//     std::cout << "Sage|Attn|" << work_items.attn_work_size() << std::endl;
//     ov::parallel_for2d_dynamic(work_items.attn_work_size(), H, [&](size_t w, size_t h) {
//         const auto& item = work_items.get_attn_work_item(w);
//         const auto batch_in_seq = item.batch_in_seq;
//         const auto batch_in_token = subsequence_begins.ptr<int32_t>()[batch_in_seq];
//         const auto q_len = item.q_len;
//         auto kv_len = static_cast<int32_t>(past_lens.ptr<int32_t>()[batch_in_seq] + q_len);
//         auto kv_block_num = static_cast<int32_t>(ov::intel_cpu::div_up(kv_len, block_size));
//         const auto past_len = past_lens.ptr<int32_t>()[batch_in_seq];
//         const auto block_id = q_len == 1 ? 0 : item.q_block_id;
//         // printf("attn w %d batch_in_seq %d q_len %d q_block_id %d\n", w, batch_in_seq, q_len, block_id);
//         ov::intel_cpu::PlainTensor sub_query;
//         sub_query.resize({static_cast<size_t>(q_len), H, S}, q.ptr<int8_t>(batch_in_token));

//         auto q_start = block_id * block_size;
//         auto q_end = std::min(q_start + block_size, q_len);
//         auto q_cnt = q_end - q_start;
//         size_t hk = h / h_each_group_len;
//         auto* q_ptr = sub_query.template ptr<int8_t>(q_start, h, 0);
//         for (int32_t k_block_id = 0; k_block_id < kv_block_num; k_block_id++) {
//             printf("ref_sage_attn\n");
//             std::vector<int32_t> temp_c(block_size * block_size, 0);
//             auto k_start = k_block_id * block_size;
//             auto k_end = std::min(k_start + block_size, kv_len);
//             auto k_cnt = k_end - k_start;
//             auto block_number =
//                 block_indices.ptr<int32_t>()[block_indices_begins.ptr<int32_t>()[batch_in_seq] + k_block_id];
//             auto* k_ptr = qk_scratch_b.ptr<int8_t>(0, k_block_id, hk);
//             bool enable_brgemm = getenv("ENABLE_BRGEMM");
//             if (q_len > 1 && enable_brgemm) {
//                 qk_s8s8_gemm[q_cnt - 1]
//                     ->executeGemm(q_cnt < block_size, q_ptr + sizeof(float), k_ptr, temp_c.data(), wsp.data(), nullptr);
//                 for (size_t i = 0; i < q_cnt; i++) {
//                     float* scale_a = reinterpret_cast<float*>(sub_query.ptr<int8_t>(q_start + i, h, 0));
//                     for (size_t j = 0; j < block_size; j++) {
//                         float* scale_b =  reinterpret_cast<float*>(qk_scratch_b.ptr<int8_t>(0, k_block_id, hk, block_size * (S - param_size)));
//                         (temp_weight.template ptr<float>(h, batch_in_token + q_start + i) + k_start)[j] =
//                             temp_c[i * block_size + j] * scale_a[0] * scale_b[j] * _d_scale;
//                         // printf(" %d ", temp_c[i * block_size + j]);
//                     }
//                     // printf("\n");
//                 }
//             } else {
//                 for (int32_t pq = 0; pq < q_cnt; pq++) {
//                     auto* q_ptr = sub_query.template ptr<int8_t>(q_start + pq, h, 0);
//                     if (k_start < kv_len) {
//                         auto* k_ptr = k_cache.ptr<int8_t, KEY_PREC>(block_number, hk);
//                         dot_product_block_s8s8_f32(
//                             q_ptr,
//                             k_ptr,
//                             temp_weight.template ptr<float>(h, batch_in_token + q_start + pq) + k_start,
//                             temp_c.data() + pq * block_size,
//                             _d_scale,
//                             S - param_size,
//                             k_cnt);
//                     }
//                 }
//                 for (size_t i = 0; i < q_cnt; i++) {
//                     float* scale_a = reinterpret_cast<float*>(sub_query.ptr<int8_t>(q_start + i, h, 0));
//                     for (size_t j = 0; j < block_size; j++) {
//                         printf(" %d ", temp_c[i * block_size + j]);
//                     }
//                     printf("\n");
//                 }
//             }
//         }
//         for (int32_t pq = 0; pq < q_cnt; pq++) {
//             auto score = temp_weight.ptr<float>(h, batch_in_token + q_start + pq);
//             auto ncausal = (past_len + q_start + pq + 1);
//             float* alibi_lookup = nullptr;
//             float alibi_slope = 0.f;
//             // if (alibi_slopes) {
//             //     alibi_slope = alibi_slopes.ptr<float>()[h];
//             //     alibi_lookup = _alibi_lookup.ptr<float>() + _alibi_lookup.m_dims[0] - ncausal;
//             // }
//             attn_softmax_kernel<float>(score,
//                                        score,
//                                        1.0f,
//                                        alibi_lookup,
//                                        nullptr,
//                                        nullptr,
//                                        false,
//                                        ncausal,
//                                        kv_len,
//                                        ov::element::f32,
//                                        ov::element::f32,
//                                        alibi_slope);
//         }
//         for (int32_t pq = 0; pq < q_cnt; pq++) {
//             for (int32_t v_block_id = 0; v_block_id < kv_block_num; v_block_id++) {
//                 auto v_start = v_block_id * block_size;
//                 auto v_end = std::min(v_start + block_size, kv_len);
//                 auto v_cnt = v_end - v_start;
//                 auto block_number =
//                     block_indices.ptr<int32_t>()[block_indices_begins.ptr<int32_t>()[batch_in_seq] + v_block_id];
//                 if (v_start < kv_len) {
//                     attn_acc_value_block_by_dim<uint8_t, ov::element::u8>(
//                         temp_output.template ptr<float>(batch_in_token + q_start + pq, h, 0),
//                         temp_weight.template ptr<float>(h, batch_in_token + q_start + pq) + v_start,
//                         v_cache.ptr<uint8_t, ov::element::u8>(block_number, hk),
//                         SV,
//                         v_cnt,
//                         SV);
//                 }
//             }
//         }
        
//         attn_memcpy2d_kernel(temp_output.ptr<float>(batch_in_token + q_start, h, 0),
//                              output_emb.ptr<DATA_TYPE>(batch_in_token + q_start, 0, h * SV),
//                              ov::element::f32,
//                              ov::intel_cpu::precision_of<DATA_TYPE>::value,
//                              temp_output.stride(0),
//                              output_emb.stride(0),
//                              SV,
//                              q_cnt);
//     });
// }

template <typename DATA_TYPE, ov::element::Type_t KEY_PREC, ov::element::Type_t VALUE_PREC>
void sage_attn(const ov::intel_cpu::PlainTensor& q,
               ov::intel_cpu::PlainTensor& k_cache,
               const ov::intel_cpu::PlainTensor& v_cache,
               const ov::intel_cpu::PlainTensor& output_emb,
               const ov::intel_cpu::PlainTensor& output_score,
               [[maybe_unused]] size_t max_context_len,
               const ov::intel_cpu::PlainTensor& past_lens,
               const ov::intel_cpu::PlainTensor& subsequence_begins,
               const ov::intel_cpu::PlainTensor& block_indices,
               const ov::intel_cpu::PlainTensor& block_indices_begins,
               const ov::intel_cpu::PlainTensor& alibi_slopes,
               ov::intel_cpu::PlainTensor& temp_weight,
               ov::intel_cpu::PlainTensor& temp_output) {
    // printf("Going to Do SageAttn\n");
    auto seq_cout = static_cast<int32_t>(past_lens.m_dims[0]);
    int32_t block_size = 32;
    size_t B = q.m_dims[0], H = q.m_dims[1], S = q.m_dims[2], SV = v_cache.m_dims[3] - sizeof(float) * 2;
    size_t Hk = k_cache.m_dims[1];
    size_t h_each_group_len = H / Hk;
    const size_t param_size = sizeof(float);
    const float _d_scale = 1 / sqrt(S - param_size);
    temp_output.resize<float>({B, H, SV});
    memset(temp_output.ptr<float>(), 0, B * H * SV * sizeof(float));
    temp_weight.resize<float>({H, B, ov::intel_cpu::rnd_up(max_context_len, std::max(block_size, int32_t{16}))});
    for (int32_t seq_id = 0; seq_id < seq_cout; seq_id++) {
        auto q_len = subsequence_begins.ptr<int32_t>()[seq_id + 1] - subsequence_begins.ptr<int32_t>()[seq_id];
        auto q_block_num = static_cast<int32_t>(ov::intel_cpu::div_up(q_len, block_size));
        const auto past_len = past_lens.ptr<int32_t>()[seq_id];
        auto kv_len = past_lens.ptr<int32_t>()[seq_id] + q_len;
        auto kv_block_num = static_cast<int32_t>(ov::intel_cpu::div_up(kv_len, block_size));
        const auto batch_in_token = subsequence_begins.ptr<int32_t>()[seq_id];
        ov::intel_cpu::PlainTensor sub_query;
        sub_query.resize({static_cast<size_t>(q_len), H, S}, q.ptr<int8_t>(batch_in_token));
        for (int32_t block_id = 0; block_id < q_block_num; block_id++) {
            // compute q_block * k_block
            auto q_start = block_id * block_size;
            auto q_end = std::min(q_start + block_size, q_len);
            auto q_cnt = q_end - q_start;
            for (size_t h = 0; h < H; h++) {
                for (size_t pq = 0; pq < q_cnt; pq++) {
                    // compute q * k
                    auto* q_ptr = sub_query.template ptr<int8_t>(q_start + pq, h, 0);
                    size_t hk = h / h_each_group_len;
                    float M = -INFINITY;  // maximum KQ value
                    float sum = 0.0f;
                    auto ncausal = (past_len + q_start + pq + 1);

                    for (int32_t k_block_id = 0; k_block_id < kv_block_num; k_block_id++) {
                        auto k_start = k_block_id * block_size;
                        auto k_end = std::min(k_start + block_size, kv_len);
                        k_end = std::min(k_end, static_cast<int32_t>(ncausal));
                        auto k_cnt = k_end - k_start;
                        if (k_start < kv_len && k_start < ncausal) {
                            auto block_number =
                                block_indices.ptr<int32_t>()[block_indices_begins.ptr<int32_t>()[seq_id] + k_block_id];
                            auto* k_ptr = k_cache.ptr<int8_t, KEY_PREC>(block_number, hk);
                            dot_product_block_s8s8_f32(
                                q_ptr,
                                k_ptr,
                                temp_weight.template ptr<float>(h, batch_in_token + q_start + pq) + k_start,
                                _d_scale,
                                S - param_size,
                                k_cnt);
                            const float Mold = M;
                            float block_max = -INFINITY;
                            for (int32_t i = 0; i < k_cnt; i++) {
                                block_max =
                                    std::max(block_max,
                                             temp_weight.at<float>({h, batch_in_token + q_start + pq, k_start + i}));
                            }
                            M = std::max(block_max, M);

                            for (int32_t i = 0; i < k_cnt; i++) {
                                temp_weight.at<float>({h, batch_in_token + q_start + pq, k_start + i}) =
                                    expf(temp_weight.at<float>({h, batch_in_token + q_start + pq, k_start + i}) - M);
                            }

                            float local_sum = 0.0f;
                            for (int32_t i = 0; i < k_cnt; i++) {
                                local_sum += temp_weight.at<float>({h, batch_in_token + q_start + pq, k_start + i});
                            }
                            float qk_scale = expf(Mold - M);

                            for (int32_t i = 0; i < SV; i++) {
                                temp_output.at<float>({batch_in_token + q_start + pq, h, i}) *= qk_scale;
                            }

                            auto* v_ptr = v_cache.ptr<uint8_t, ov::element::u8>(block_number, hk);
                            attn_acc_value_block_by_dim<uint8_t, ov::element::u8>(
                                temp_output.template ptr<float>(batch_in_token + q_start + pq, h, 0),
                                temp_weight.template ptr<float>(h, batch_in_token + q_start + pq) + k_start,
                                v_ptr,
                                SV,
                                k_cnt,
                                SV);
                            sum = sum * qk_scale + local_sum;
                        }
                    }
                    float inv_sum = 1.0f / sum;
                    for (size_t i = 0; i < SV; i++) {
                        temp_output.at<float>({batch_in_token + q_start + pq, h, i}) *= inv_sum;
                    }
                }
                attn_memcpy2d_kernel(temp_output.ptr<float>(batch_in_token + q_start, h, 0),
                                     output_emb.ptr<DATA_TYPE>(batch_in_token + q_start, 0, h * SV),
                                     ov::element::f32,
                                     ov::element::bf16,
                                     temp_output.stride(0),
                                     output_emb.stride(0),
                                     SV,
                                     q_cnt);
            }
        }
    }
}

}  // namespace ov::Extensions::Cpu::XARCH