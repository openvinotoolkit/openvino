// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_attn.h"

#include <dnnl_extension_utils.h>
#include <onednn/dnnl.h>

#include <chrono>
#include <algorithm>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <ie_ngraph_utils.hpp>
#include <string>
#include <shape_inference/shape_inference_internal_dyn.hpp>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include "emitters/x64/jit_dnnl_emitters.hpp"
#include "emitters/x64/jit_load_store_emitters.hpp"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "utils/plain_tensor.hpp"
#include <openvino/op/scaled_dot_product_attention.hpp>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <immintrin.h>
#endif

#ifdef OV_CPU_WITH_MLAS
#    include "mlas/sgemm.hpp"
#endif

#include "utils/plain_tensor.hpp"
#include "utils/profiler.hpp"
#include "scaled_attn_softmax.hpp"

using namespace InferenceEngine;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

// TODO: profile
#define PROFILE(prof, name)
#define PROFILE_NEXT(prof, name)


namespace ov {
namespace intel_cpu {
namespace node {

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "

//============================ kernels ============================
enum KernelTypes { KT_REF, KT_LLMDNN, KT_MLAS};

template <KernelTypes KType>
struct ktype_name_of {
    static constexpr const char* value = "?";
};

template <>
struct ktype_name_of<KT_REF> {
    static constexpr const char* value = "REF";
};
template <>
struct ktype_name_of<KT_LLMDNN> {
    static constexpr const char* value = "LLMDNN";
};
template <>
struct ktype_name_of<KT_MLAS> {
    static constexpr const char* value = "MLAS";
};

// default implementation: reference
template <KernelTypes KType, typename T>
struct MHA_kernel {
    MHA_kernel() = default;

    template <typename D>
    float dot_product(const D* a, const D* b, int len, int stride_b = 1) {
        float result = 0;
        if (stride_b == 1) {
            for (int i = 0; i < len; i++)
                result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
        } else {
            for (int i = 0; i < len; i++)
                result += static_cast<float>(a[i]) * static_cast<float>(b[i * stride_b]);
        }
        return result;
    }

    void softmax(float* a, int len) {
        float max = *std::max_element(a, a + len);
        float sum = 0.0f;
        for (int i = 0; i < len; i++) {
            a[i] = exp(a[i] - max);
            sum += a[i];
        }
        float scale = 1.0f / sum;
        for (int i = 0; i < len; i++) {
            a[i] *= scale;
        }
    }

    template <typename D>
    void accumulate(float* acc, const D* v, int len, float weight = 1.0f) {
        for (int i = 0; i < len; i++) {
            acc[i] += static_cast<float>(v[i]) * weight;
        }
    }

    PlainTensor<uint8_t> causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor<uint8_t> mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // output_emb    [B, q_len, H*S]
    void operator()(PlainTensor<T>& query,
                    PlainTensor<T>& present_key,
                    PlainTensor<T>& present_value,
                    const PlainTensor<float>& alibi_mask,
                    const PlainTensor<float>& attention_mask,
                    PlainTensor<T>& output_emb,
                    bool has_out_transpose,
                    float d_scale = 0.0f) {
        PROFILE(prof, "MHA_REF");
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);

        auto k_stride_s = present_key.stride(3);
        bool auto_causal = attention_mask.size(2) == 1 && !causal_mask;

        parallel_for2d(B, H, [&](size_t b, size_t h) {
            std::vector<float> attn_score(kv_len);
            std::vector<float> word_vec(head_size, 0.0f);

            // auto key = &present_key.at({b, h, 0, 0});
            // auto value = &present_value.at({b, h, 0, 0});
            // auto output = &output_emb.at({b, 0, h * head_size});
            for (size_t m = 0; m < q_len; m++) {
                // dot-product to get attention scores
                auto* q = &query.at({b, h, m, 0});
                // how many key/values can be accessed causally
                auto ncausal = kv_len;  // kv_len - q_len + m + 1;
                // no causall mask is set and it's not fused into attention_mask
                if (auto_causal)
                    ncausal = kv_len - q_len + m + 1;
                for (size_t n = 0; n < ncausal; n++) {
                    auto* k = &present_key.at({b, h, n, 0});
                    attn_score[n] = dot_product(q, k, head_size, k_stride_s) * d_scale;

                    // apply alibi tensor
                    if (alibi_mask)
                        attn_score[n] += alibi_mask.at({b, h, m, n}, true);

                    // apply attention mask (maybe combined with causal_mask)
                    if (attention_mask)
                        attn_score[n] += attention_mask.at({b, h, m, n}, true);

                    // apply causal_mask
                    if (causal_mask) {
                        bool is_zero = causal_mask.at({b, h, m, n}, true) == 0;
                        if (select_nfltmax_at_0) {
                            if (is_zero)
                                attn_score[n] = -FLT_MAX;
                        } else {
                            if (!is_zero) {
                                attn_score[n] = -FLT_MAX;
                            }
                        }
                    }
                }

                // softmax
                softmax(&attn_score[0], ncausal);

                // linearly combine value
                word_vec.assign(head_size, 0.0f);
                for (size_t n = 0; n < ncausal; n++) {
                    auto* v = &present_value.at({b, h, n, 0});
                    accumulate(word_vec.data(), v, head_size, attn_score[n]);
                }

                // output [B, L1, H*head_size]
                auto* out = has_out_transpose ? &output_emb.at({b, m, h * head_size}) : &output_emb.at({b, h, m});
                std::copy(word_vec.begin(), word_vec.end(), out);
            }
        });
    }
};

#ifdef OV_CPU_WITH_MLAS
template <>
struct MHA_kernel<KT_MLAS, float> {
    size_t m_block_size;
    // buffer to hold qk temp
    std::vector<PlainTensor<float>> qk_buffers;

    MHA_kernel() {
        m_block_size = std::getenv("MBLK") ? atoi(std::getenv("MBLK")) : 4;
        qk_buffers.resize(parallel_get_max_threads(), PlainTensor<float>(true));
    }

    PlainTensor<uint8_t> causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor<uint8_t> mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // alibi
    // output_emb    [B, L1, H*S]
    void operator()(PlainTensor<float>& query,
                    PlainTensor<float>& present_key,
                    PlainTensor<float>& present_value,
                    const PlainTensor<float>& alibi_mask,
                    const PlainTensor<float>& attention_mask,
                    PlainTensor<float>& output_emb,
                    bool has_out_transpose,
                    float d_scale = 0.0f) {
        PROFILE(prof, "MHA_MLAS");
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);
        auto h_group_num = present_key.size(1);
        size_t h_each_group_len = H / h_group_num;

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);
        auto k_stride_s = present_key.stride(3);

        auto m_blocks = (q_len + m_block_size - 1) / m_block_size;
        bool auto_causal = attention_mask.size(2) == 1 && !causal_mask;

        parallel_for3d(B, H, m_blocks, [&](size_t b, size_t h, size_t m_blk) {
            size_t thread_id = static_cast<size_t>(parallel_get_thread_num());
            auto& qk_buf = qk_buffers[thread_id];

            auto m_start = m_blk * m_block_size;
            auto m_end = std::min(m_start + m_block_size, q_len);
            auto m_cnt = m_end - m_start;

            auto kv_len_cache_align = (((kv_len * sizeof(float)) + 63) / 64 * 64) / sizeof(float);
            qk_buf.resize({m_block_size, kv_len_cache_align});
            const float* q_ptr = &query.at({b, h, m_start, 0});
            const float* k_ptr = &present_key.at({b, h / h_each_group_len, 0, 0});
            const float* v_ptr = &present_value.at({b, h / h_each_group_len, 0, 0});

            float* alibi_ptr = nullptr;
            auto alibi_stride = 0;
            if (alibi_mask) {
                alibi_ptr = &alibi_mask.at({b, h, 0, 0}, true);
                if (alibi_mask.size(2) > 1)
                    alibi_stride = alibi_mask.stride(2);
            }
            float* attn_mask_ptr = nullptr;
            auto attn_mask_stride = 0;
            if (attention_mask) {
                attn_mask_ptr = &attention_mask.at({b, h, 0, 0}, true);
                if (attention_mask.size(2) > 1)
                    attn_mask_stride = attention_mask.stride(2);
            }
            uint8_t* cmask_ptr = nullptr;
            auto cmask_stride = 0;
            if (causal_mask) {
                cmask_ptr = &causal_mask.at({b, h, 0, 0}, true);
                if (causal_mask.size(2) > 1)
                    cmask_stride = causal_mask.stride(2);
            }

            float* qk = &(qk_buf.at({0, 0}));
            auto qk_m_stride = qk_buf.stride(0);

            if (k_stride_s == 1)
                mlas_sgemm("N",
                           "T",
                           m_cnt,
                           kv_len,
                           head_size,
                           1.0f,
                           q_ptr,
                           query.stride(2),
                           k_ptr,
                           present_key.stride(2),
                           0.f,
                           qk,
                           qk_m_stride,
                           1);
            else
                mlas_sgemm("N",
                           "N",
                           m_cnt,
                           kv_len,
                           head_size,
                           1.0f,
                           q_ptr,
                           query.stride(2),
                           k_ptr,
                           present_key.stride(3),
                           0.f,
                           qk,
                           qk_m_stride,
                           1);

            for (size_t m = m_start; m < m_end; m++) {
                // apply attention mask & sofmax
                auto ncausal = auto_causal ? (kv_len - q_len + m + 1) : kv_len;
                InferenceEngine::Extensions::Cpu::XARCH::attn_softmax(qk + (m - m_start) * qk_m_stride,
                                                                      d_scale,
                                                                      alibi_ptr + m * alibi_stride,
                                                                      attn_mask_ptr + m * attn_mask_stride,
                                                                      cmask_ptr + m * cmask_stride,
                                                                      select_nfltmax_at_0,
                                                                      ncausal,
                                                                      kv_len);
            }
            mlas_sgemm("N",
                       "N",
                       m_cnt,
                       head_size,
                       kv_len,
                       1.0f,
                       qk,
                       qk_m_stride,
                       v_ptr,
                       present_value.stride(2),
                       0.f,
                       has_out_transpose ? &output_emb.at({b, m_start, h * head_size}) : &output_emb.at({b, h, m_start}),
                       has_out_transpose ? output_emb.stride(1) : output_emb.stride(2),
                       1);
        });
    }
};
#endif

#ifdef __AVX512F__
#define ENABLE_AVX512_OPT
#endif

// 2nd token case : only 1 token in query
template <typename RT>
struct MHA_1Token {
    PlainTensor<float> m_attn_w;
    PlainTensor<float> m_temp;

    MHA_1Token() : m_temp(true), m_attn_w(true) {}

    PlainTensor<uint8_t> causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor<uint8_t> mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // alibi
    // output_emb    [B, L1, H*S]
    void operator()(PlainTensor<RT>& query,
                    PlainTensor<RT>& present_key,
                    PlainTensor<RT>& present_value,
                    const PlainTensor<float>& alibi_mask,
                    const PlainTensor<float>& attention_mask,
                    PlainTensor<RT>& output_emb,
                    const PlainTensor<int32_t>& beams,
                    bool has_out_transpose,
                    float d_scale = 0.0f) {
        PROFILE(prof0, "MHA_1Token");
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto S = query.size(3);
        auto kv_len = present_key.size(2);
        auto h_group_num = present_key.size(1);
        size_t h_each_group_len = 0;
        if (h_group_num != H) {
            h_each_group_len = H / h_group_num;
        }

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(S);
        auto k_stride_s = present_key.stride(3);

        assert(k_stride_s == 1);

        bool auto_causal = attention_mask.size(2) == 1 && !causal_mask;

        PROFILE(prof, "Q*K");
        // use per-token kernel, for each k,v token
        //  attn mask is a matrix of q_len(kv_len)
        m_attn_w.resize({B, H, q_len, kv_len});

        if (h_each_group_len) {
            parallel_for3d(B, H / h_each_group_len, kv_len, [&](size_t b, size_t h_group, size_t pk) {
                // which batch item should be used at postion pk?
                auto b_kv = beams ? beams.at({b, pk}) : b;
                for (size_t pq = 0; pq < q_len; pq++) {
                    for (size_t h = h_group * h_each_group_len; h < (h_group + 1) * h_each_group_len; h++) {
                        auto sum = dot_product_opt(&query.at({b, h, pq, 0}), &present_key.at({b_kv, h_group, pk, 0}), S);
                        m_attn_w.at({b, h, pq, pk}) = sum;
                    }
                }
            });
        } else {
            parallel_for3d(B, H, kv_len, [&](size_t b, size_t h, size_t pk) {
                // which batch item should be used at postion pk?
                auto b_kv = beams ? beams.at({b, pk}) : b;
                for (size_t pq = 0; pq < q_len; pq++) {
                    auto sum = dot_product_opt(&query.at({b, h, pq, 0}), &present_key.at({b_kv, h, pk, 0}), S);
                    m_attn_w.at({b, h, pq, pk}) = sum;
                }
            });
        }

        PROFILE_NEXT(prof, "softmax");

        parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
            // apply attention mask & sofmax
            auto ncausal = auto_causal ? (kv_len - q_len + pq + 1) : kv_len;
            float* alibi_ptr = alibi_mask ? &alibi_mask.at({b, h, pq, 0}, true) : nullptr;
            float* attn_mask_ptr = attention_mask ? &attention_mask.at({b, h, pq, 0}, true) : nullptr;
            uint8_t* cmask_ptr = causal_mask ? &causal_mask.at({b, h, pq, 0}, true) : nullptr;
            InferenceEngine::Extensions::Cpu::XARCH::attn_softmax(&m_attn_w.at({b, h, pq, 0}),
                                                                  d_scale,
                                                                  alibi_ptr,
                                                                  attn_mask_ptr,
                                                                  cmask_ptr,
                                                                  select_nfltmax_at_0,
                                                                  ncausal,
                                                                  kv_len);
        });

        PROFILE_NEXT(prof, "W*V");
        // attn_w * V
        auto nthr = parallel_get_max_threads();
        m_temp.resize({static_cast<size_t>(nthr), B, q_len, H, S});
        // m_attn_w {B, H, q_len, kv_len}
        if (h_each_group_len == 0) {
            parallel_nt_static(nthr, [&](const size_t ithr, const size_t nthr) {
                size_t start{0}, end{0};
                splitter(B * H * kv_len, nthr, ithr, start, end);

                memset(&m_temp.at({ithr, 0, 0, 0, 0}), 0, m_temp.stride(0) * sizeof(float));

                size_t b, h, pv;
                parallel_it_init(start, b, B, h, H, pv, kv_len);
                for (size_t iwork = start; iwork < end; ++iwork) {
                    auto b_kv = beams ? beams.at({b, pv}) : b;
                    auto* v = &present_value.at({b_kv, h, pv, 0});
                    for (size_t pq = 0; pq < q_len; pq++) {
                        auto* out = &m_temp.at({ithr, b, pq, h, 0});
                        auto weight = m_attn_w.at({b, h, pq, pv});
                        accumulate_weighted_v(out, weight, v, S);
                    }
                    parallel_it_step(b, B, h, H, pv, kv_len);
                }
            });
        } else {
            parallel_nt_static(nthr, [&](const size_t ithr, const size_t nthr) {
                size_t start{0}, end{0};
                splitter(B * h_group_num * kv_len, nthr, ithr, start, end);

                memset(&m_temp.at({ithr, 0, 0, 0, 0}), 0, m_temp.stride(0) * sizeof(float));

                size_t b, h_group, pv;
                parallel_it_init(start, b, B, h_group, h_group_num, pv, kv_len);
                for (size_t iwork = start; iwork < end; ++iwork) {
                    auto b_kv = beams ? beams.at({b, pv}) : b;
                    auto* v = &present_value.at({b_kv, h_group, pv, 0});
                    for (size_t pq = 0; pq < q_len; pq++) {
                        for (size_t h = h_group * h_each_group_len; h < (h_group + 1) * h_each_group_len; h++) {
                            auto* out = &m_temp.at({ithr, b, pq, h, 0});
                            auto weight = m_attn_w.at({b, h, pq, pv});
                            accumulate_weighted_v(out, weight, v, S);
                        }
                    }
                    parallel_it_step(b, B, h_group, h_group_num, pv, kv_len);
                }
            });
        }

        PROFILE_NEXT(prof, "Reduce");
        parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
            auto* temp = &m_temp.at({0, b, pq, h, 0});
            size_t temp_stride = m_temp.stride(0);
            auto* dst = has_out_transpose ? &output_emb.at({b, pq, h*S}) : &output_emb.at({b, h, pq});
            reduce_v(dst, temp, nthr, S, temp_stride);
        });
    }

    template <bool trans_B = false, class TC, class TA, class TB>
    void matmul(size_t M, size_t N, size_t K, TC* C, size_t ldC, TA* A, size_t ldA, TB* B, size_t ldB) {
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                auto& sum = C[m * ldC + n];
                sum = 0;
                for (size_t k = 0; k < K; k++) {
                    if (trans_B) {
                        sum += A[m * ldA + k] * B[n * ldB + k];
                    } else {
                        sum += A[m * ldA + k] * B[k * ldB + n];
                    }
                }
            }
        }
    }

    inline __m512 mm512_uni_loadu_ps(ov::bfloat16* a) {
#ifdef __AVX512BF16__
        auto vec_bf16 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a));
        __m512i y = _mm512_cvtepu16_epi32(vec_bf16);
        return _mm512_castsi512_ps(_mm512_slli_epi32(y, 16));
#endif
    }
    inline __m512 mm512_uni_loadu_ps(float* a) {
        return _mm512_loadu_ps(a);
    }
    inline void mm512_uni_storeu_ps(float* a,  __m512 v) {
        _mm512_storeu_ps(a, v);
    }
    inline void mm512_uni_storeu_ps(ov::bfloat16* a,  __m512 v) {
#ifdef __AVX512BF16__
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a),
                            reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(v)));
#endif
    }

#ifdef __AVX2__
    inline __m256 mm256_uni_loadu_ps(float* a) {
        return _mm256_loadu_ps(a);
    }
    inline void mm256_uni_storeu_ps(float* a,  __m256 v) {
        _mm256_storeu_ps(a, v);
    }

    inline __m256 mm256_uni_loadu_ps(ov::bfloat16* a) {
        auto vec_bf16 = _mm_loadu_si128(reinterpret_cast<__m128i*>(a));
        auto o = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(vec_bf16), 16));
        return o;
    }

    inline void mm256_uni_storeu_ps(ov::bfloat16* a,  __m256 v) {
        __m256i iv = _mm256_castps_si256(v);
        __m256i nan = _mm256_set1_epi32(0xffff);
        __m256i mask = _mm256_castps_si256(_mm256_cmp_ps(v, v, _CMP_ORD_Q));
        __m256i ones = _mm256_set1_epi32(0x00010000);;
        // uint32_t int_i =  input & 1;
        auto int_i = _mm256_and_si256(iv, ones);
        int_i = _mm256_srli_epi32(int_i, 1);
        int_i = _mm256_srli_epi32(_mm256_add_epi32(int_i, iv), 16);
        int_i = _mm256_blendv_epi8(nan, int_i, mask);
        int_i = _mm256_packus_epi32(int_i, int_i);
        int_i = _mm256_permute4x64_epi64(int_i, 0xd8);
        __m128i bf16_o = _mm256_extractf128_si256(int_i, 1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(a),
                    reinterpret_cast<__m128i>(bf16_o));
    }

inline void hsum(__m256& x) {
    __m256 y;                             // x:  0 1 2 3   4 5 6 7
    y = _mm256_permute_ps(x, 0x39);       // y:  1 2 3 0   5 6 7 4
    x = _mm256_add_ps(x, y);              // X:  01 12 23 30  45 56 67 74
    y = _mm256_permute_ps(x, 0x4e);       // y:  23 30 01 12  67 74 45 56
    x = _mm256_add_ps(x, y);              // x: 0123 x x x   4567 x x x
    y = _mm256_permute2f128_ps(x, x, 1);  // y: 4567 x x x  0123 x x x
    x = _mm256_add_ps(x, y);              // x: 01234567 x x x x x x x
}
#endif


    template<typename T>
    float dot_product_opt(T* a, T* b, size_t n) {
        size_t i = 0;
        float sum = 0.0f;
#ifdef ENABLE_AVX512_OPT
        auto vsum = _mm512_setzero_ps();
        for (; i <= n - 16; i += 16) {
            auto va = mm512_uni_loadu_ps(a + i);
            auto vb = mm512_uni_loadu_ps(b + i);
            vsum = _mm512_fmadd_ps(va, vb, vsum);
        }
        sum = _mm512_reduce_add_ps(vsum);
#elif defined(__AVX2__)
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

    template<typename TO, typename TI>
    void accumulate_weighted_v(TO* out, float weight, TI* v, size_t S) {
        size_t i = 0;
#ifdef ENABLE_AVX512_OPT
        auto attn_w_vec_fp32 = _mm512_set1_ps(weight);
        for (; i <= S - 16; i +=16) {
            auto v_value = mm512_uni_loadu_ps(v + i);
            auto v_out = mm512_uni_loadu_ps(out + i);
            v_out = _mm512_fmadd_ps(attn_w_vec_fp32, v_value, v_out);
            _mm512_storeu_ps(out + i, v_out);
        }
#elif defined(__AVX2__)
        auto attn_w_vec_fp32 = _mm256_set1_ps(weight);
        for (; i <= S - 8; i += 8) {
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

    template<typename T>
    void reduce_v(T* dst, float* temp, size_t M, size_t S, size_t temp_stride) {
        size_t i = 0;
#ifdef ENABLE_AVX512_OPT
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
#elif defined(__AVX2__)
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
};

template <KernelTypes KType, typename T>
struct AttentionExecutor : public ScaledDotProductAttention::Executor {
    PlainTensor<T> q_input;           // f32[B, L1, H*S] / [B, H, L1, S]
    PlainTensor<T> k_input;           // f32[B, L1, H*S]
    PlainTensor<T> v_input;           // f32[B, L1, H*S]
    PlainTensor<T> k_cache;           // f32[B, H, max_kvLen, S]
    PlainTensor<T> v_cache;           // f32[B, H, max_kvLen, S]
    PlainTensor<int32_t> beam_table;  // i32[B, max_kvLen]
    PlainTensor<float> attn_mask;     // f32[B, qLen + kvLen]
    PlainTensor<float> cos_tab;       // f32[max_kv_len, rotary_dims//2]
    PlainTensor<float> sin_tab;       // f32[max_kv_len, rotary_dims//2]

    PlainTensor<T> output_emb;        // f32[B, L1, H*S]

    MHA_kernel<KType, T> kernel;
    MHA_1Token<T> kernel_1tok;

    PlainTensor<T> m_query_emb;  // query with RoPE position embedding

    void execute(ScaledDotProductAttention* node) override {
        q_input.reset(node->getParentEdgeAt(0)->getMemoryPtr());
        k_input.reset(node->getParentEdgeAt(1)->getMemoryPtr());
        v_input.reset(node->getParentEdgeAt(2)->getMemoryPtr());
        attn_mask.reset(node->getParentEdgeAt(3)->getMemoryPtr());     // f32[1, 1, B, qLen + kvLen]
        bool has_out_transpose = node->is_out_transpose();
        auto rope_type = node->get_rope_type();
        bool is_causal = node->is_causal();

        size_t B, H, L1, L0, S;
        if (rope_type != -1) {
            k_cache.reset(node->getParentEdgeAt(4)->getMemoryPtr());
            v_cache.reset(node->getParentEdgeAt(5)->getMemoryPtr());
            cos_tab.reset(node->getParentEdgeAt(6)->getMemoryPtr());
            sin_tab.reset(node->getParentEdgeAt(7)->getMemoryPtr());
            // q: [B, L1, H*S]
            B = q_input.size(0);
            L1 = q_input.size(1);
            L0 = attn_mask.size(1) - L1;
            S = k_cache.size(-1);
            // TODO: get H?
        } else {
            // q, k, v: [B, H, L0, S]
            B = q_input.size(0);
            H = q_input.size(1);
            L1 = q_input.size(2);
            L0 = k_input.size(2) - L1;
            S = q_input.size(-1);
        }

        attn_mask.assert_dims({1, 1, B, L0 + L1});
        attn_mask = attn_mask.reshape({B, 1, 1, L0 + L1});

        {
            PROFILE(prof, "redefineOutputMemory");
            if (has_out_transpose)
                node->redefineOutputMemory({{B, L1, H * S}});
            else
                node->redefineOutputMemory({{B, H, L1, S}});
        }
        ov::intel_cpu::PlainTensor<T> output_emb(node->getChildEdgeAt(0)->getMemoryPtr());
        PlainTensor<T> present_key, present_value;

        if (rope_type != -1) {
            // auto layer_id = static_cast<int>(attr_map["layer_id"]);
            // auto rotary_dims = static_cast<int>(attr_map["rotary_dims"]);
            // auto rope_type = static_cast<int>(attr_map["rope_type"]);
            // auto num_kv_heads = static_cast<size_t>(attr_map["num_kv_heads"]);
            auto half_rotary_dims = cos_tab.size(-1);
            cos_tab.assert_dims({0, half_rotary_dims}, true);
            sin_tab.assert_dims({0, half_rotary_dims}, true);

            q_input.assert_dims({B, L1, H * S});
            k_input.assert_dims({B, L1, H * S});
            v_input.assert_dims({B, L1, H * S});

            auto rope_q = q_input.reshape({B, L1, H, S});
            auto rope_k = k_input.reshape({B, L1, H, S});
            auto rope_v = v_input.reshape({B, L1, H, S});

            // kv cache is just a partial view of a big buffer
            m_query_emb.resize({B, H, L1, S});

            present_key = k_cache.index({{0, static_cast<int>(B)},
                                            {0, static_cast<int>(H)},
                                            {0, static_cast<int>(L0 + L1)},
                                            {}});
            present_value = v_cache.index({{0, static_cast<int>(B)},
                                                {0, static_cast<int>(H)},
                                                {0, static_cast<int>(L0 + L1)},
                                                {}});

            //half_rotary_dims = rotary_dims / 2;
            auto rotary_dims = half_rotary_dims * 2;

            parallel_for3d(B, H, L1, [&](size_t b, size_t h, size_t p) {
                auto p1 = p + L0;
                size_t position_id = p1;
                /*
                // position derived from attention mask
                // needs to skip where attention < 0
                // but it's not required when padding at left
                for (size_t i = 0; i < p1; i++) {
                    if (attn_mask.at({b, 0, 0, i}) >= 0.0f) {
                        position_id++;
                    }
                }
                */
                auto* q_embed = &m_query_emb.at({b, h, p, 0});
                auto* cos = &cos_tab({position_id, 0});
                auto* sin = &sin_tab({position_id, 0});
                T* q;
                T* k;
                T* v;

                size_t g = h; // gH;
                size_t hg = 0;//h % gH;
                q = &rope_q.at({b, p, h, 0});
                k = &rope_k.at({b, p, h, 0});
                v = &rope_v.at({b, p, h, 0});
                auto* present_k = &present_key.at({b, g, p1, 0});    // f32[B, H, L0+L1, 64]
                auto* present_v = &present_value.at({b, g, p1, 0});  // f32[B, H, L0+L1, 64]

                size_t s = 0;
                if (rope_type > 0) {
                    // gptneox RoPE
                    for (size_t i = 0; s < half_rotary_dims; i++, s++) {
                        q_embed[s] = cos[i] * q[s] + sin[i] * (-q[s + half_rotary_dims]);
                        if (hg == 0) {
                            present_k[s] = cos[i] * k[s] + sin[i] * (-k[s + half_rotary_dims]);
                            present_v[s] = v[s];
                        }
                    }
                    for (size_t i = 0; s < rotary_dims; i++, s++) {
                        q_embed[s] = cos[i] * q[s] + sin[i] * (q[i]);
                        if (hg == 0) {
                            present_k[s] = cos[i] * k[s] + sin[i] * (k[i]);
                            present_v[s] = v[s];
                        }
                    }
                } else {
                    // gptj RoPE
                    for (size_t i = 0; s < rotary_dims; i++, s += 2) {
                        q_embed[s] = cos[i] * q[s] - sin[i] * q[s + 1];
                        q_embed[s + 1] = cos[i] * q[s + 1] + sin[i] * q[s];

                        if (hg == 0) {
                            present_k[s] = cos[i] * k[s] - sin[i] * k[s + 1];
                            present_k[s + 1] = cos[i] * k[s + 1] + sin[i] * k[s];

                            present_v[s] = v[s];
                            present_v[s + 1] = v[s + 1];
                        }
                    }
                }

                for (; s < S; s++) {
                    q_embed[s] = q[s];
                    if (hg == 0) {
                        present_k[s] = k[s];
                        present_v[s] = v[s];
                    }
                }
            });
        } else {
            q_input.assert_dims({B, H, L1, S});
            k_input.assert_dims({B, H, L0 + L1, S});
            v_input.assert_dims({B, H, L0 + L1, S});
            m_query_emb = q_input;
            present_key = k_input;
            present_value = v_input;
        }

        if (L1 > 1) {
            // multi-token version
            kernel(m_query_emb, present_key, present_value, {}, attn_mask, output_emb, has_out_transpose);
        } else {
            // 1-token version
            kernel_1tok(m_query_emb, present_key, present_value, {}, attn_mask, output_emb, beam_table, has_out_transpose);
        }
    }
};

ScaledDotProductAttention::ScaledDotProductAttention(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    const auto node = std::dynamic_pointer_cast<const ov::op::v12::ScaledDotProductAttention>(op);
    m_is_causal = node->get_is_causal();
}

void ScaledDotProductAttention::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    auto srcPrecision = getOriginalInputPrecisionAtPort(0);
    auto dstPrecision = getOriginalOutputPrecisionAtPort(0);

    auto rtPrecision = InferenceEngine::Precision::FP32; // srcPrecision;

    if (rtPrecision == InferenceEngine::Precision::BF16) {
        m_executor = std::make_shared<AttentionExecutor<KT_REF, ov::bfloat16>>();
    } else {
        m_executor = std::make_shared<AttentionExecutor<KT_MLAS, float>>();
    }

    // initialize input ports
    std::vector<PortConfigurator> inPortConfigs;
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(1), false, -1);
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(2), false, -1);
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(3), false, -1);

    // initialize output port
    std::vector<PortConfigurator> outPortConfigs;
    outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void ScaledDotProductAttention::execute(dnnl::stream strm) {
    m_executor->execute(this);
}

bool ScaledDotProductAttention::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto node = std::dynamic_pointer_cast<const ov::op::v12::ScaledDotProductAttention>(op);
        if (!node) {
            errorMessage = "Only ScaledDotProductAttention operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
