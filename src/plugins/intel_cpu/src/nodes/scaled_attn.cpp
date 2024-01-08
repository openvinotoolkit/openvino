// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_attn.h"

#include "common/arbitrary_order_desc_creator.h"
#include "common/primitive_hashing_utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/util/common_util.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/plain_tensor.hpp"

#ifdef OV_CPU_WITH_MLAS
#    include "mlas/sgemm.hpp"
#endif

#include "utils/plain_tensor.hpp"
#include "kernels/scaled_attn/softmax.hpp"
#include "kernels/scaled_attn/mha_single_token.hpp"
#include "kernels/scaled_attn/attn_memcpy.hpp"

#include <algorithm>
#include <string>
#include <vector>

using namespace ov::Extensions::Cpu::XARCH;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {
namespace node {

struct ScaledDotProductAttentionKey {
    ov::element::Type rtPrecision;

    size_t hash() const;
    bool operator==(const ScaledDotProductAttentionKey& rhs) const;
};

size_t ScaledDotProductAttentionKey::hash() const {
    size_t seed = 0;
    seed = hash_combine(seed, rtPrecision.hash());

    return seed;
}

bool ScaledDotProductAttentionKey::operator==(const ScaledDotProductAttentionKey& rhs) const {
    auto retVal = rtPrecision == rhs.rtPrecision;

    return retVal;
}

// default implementation: reference
template <ScaledDotProductAttention::KernelTypes KType, typename T>
struct MHAKernel {
    MHAKernel() = default;

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

    PlainTensor causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // output_emb    [B, q_len, H*S]
    void operator()(dnnl::stream strm,
                    PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    const PlainTensor& alibi_mask,
                    const PlainTensor& attention_mask,
                    PlainTensor& output_emb,
                    bool has_out_transpose,
                    bool auto_causal,
                    float d_scale = 0.0f) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);

        auto k_stride_s = present_key.stride(3);

        parallel_for2d(B, H, [&](size_t b, size_t h) {
            std::vector<float> attn_score(kv_len);
            std::vector<float> word_vec(head_size, 0.0f);

            for (size_t m = 0; m < q_len; m++) {
                // dot-product to get attention scores
                auto* q = &query.at<T>({b, h, m, 0});
                // how many key/values can be accessed causally
                auto ncausal = kv_len;
                // no causall mask is set and it's not fused into attention_mask
                if (auto_causal)
                    ncausal = kv_len - q_len + m + 1;
                for (size_t n = 0; n < ncausal; n++) {
                    auto* k = &present_key.at<T>({b, h, n, 0}, true);
                    attn_score[n] = dot_product(q, k, head_size, k_stride_s) * d_scale;

                    // apply alibi tensor
                    if (alibi_mask)
                        attn_score[n] += alibi_mask.at<float>({b, h, m, n}, true);

                    // apply attention mask (maybe combined with causal_mask)
                    if (attention_mask)
                        attn_score[n] += attention_mask.at<float>({b, h, m, n}, true);

                    // apply causal_mask
                    if (causal_mask) {
                        bool is_zero = causal_mask.at<uint8_t>({b, h, m, n}, true) == 0;
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
                    auto* v = &present_value.at<T>({b, h, n, 0}, true);
                    accumulate(word_vec.data(), v, head_size, attn_score[n]);
                }

                // output [B, L1, H*head_size]
                auto* out = has_out_transpose ? &output_emb.at<T>({b, m, h * head_size}) : &output_emb.at<T>({b, h, m});
                std::copy(word_vec.begin(), word_vec.end(), out);
            }
        });
    }
};

template <typename T>
struct MHAKernel<ScaledDotProductAttention::KT_ONEDNN, T> {
    // q: [B, H, q_len, S]
    // k: [B, H, kv_len, S]
    // v: [B, H, kv_len, S]
    dnnl::memory::desc q_md;
    dnnl::memory::desc k_md;
    dnnl::memory::desc weight_md;
    dnnl::memory::desc v_md;
    dnnl::memory::desc out_md;
    dnnl::memory attn_score;
    dnnl::memory attn_weight;
    dnnl::matmul qk_prim;
    dnnl::matmul wv_prim;
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;

    void prepare_prim(dnnl::stream strm,
                      PlainTensor& query,
                      PlainTensor& present_key,
                      PlainTensor& present_value,
                      size_t B,
                      size_t H,
                      size_t Hk,
                      size_t q_len,
                      size_t kv_len,
                      size_t S,
                      bool has_out_transpose) {
        auto make_dnnl_dims = [](const std::vector<size_t>& dims) {
            dnnl::memory::dims dnnl_dims(dims.size());
            for (size_t i = 0; i < dims.size(); i++)
                dnnl_dims[i] = static_cast<dnnl::memory::dim>(dims[i]);
            return dnnl_dims;
        };
        auto qkv_dt = precision_of<T>::value == ov::element::f32 ? dt::f32 : dt::bf16;
        dnnl::memory::desc cur_q_md(make_dnnl_dims({B, H, q_len, S}), qkv_dt, query.get_strides<dnnl::memory::dim>());
        dnnl::memory::desc cur_k_md(make_dnnl_dims({B, Hk, kv_len, S}), qkv_dt, present_key.get_strides<dnnl::memory::dim>());
        if (cur_q_md == q_md && cur_k_md == k_md)
            return;

        q_md = cur_q_md;
        k_md = cur_k_md;
        dnnl::memory::desc attn_md(make_dnnl_dims({B, H, q_len, kv_len}), dt::f32, tag::abcd);
        k_md = k_md.permute_axes({0, 1, 3, 2});
        auto qk_pd = dnnl::matmul::primitive_desc(strm.get_engine(), q_md, k_md, attn_md);
        qk_prim = dnnl::matmul(qk_pd);

        weight_md = dnnl::memory::desc(make_dnnl_dims({B, H, q_len, kv_len}), qkv_dt, tag::abcd);
        v_md = dnnl::memory::desc(make_dnnl_dims({B, Hk, kv_len, S}), qkv_dt, present_value.get_strides<dnnl::memory::dim>());
        out_md = dnnl::memory::desc(make_dnnl_dims({B, H, q_len, S}), qkv_dt, tag::abcd);
        if (has_out_transpose)
            out_md = out_md.permute_axes({0, 2, 1, 3});
        auto wv_pd = dnnl::matmul::primitive_desc(strm.get_engine(), weight_md, v_md, out_md);
        wv_prim = dnnl::matmul(wv_pd);

        if (!attn_score || attn_md.get_size() > attn_score.get_desc().get_size()) {
            attn_score = dnnl::memory(attn_md, strm.get_engine());
            attn_weight = dnnl::memory(weight_md, strm.get_engine());
        }
    }

    void exec_qk(dnnl::stream strm, PlainTensor& query, PlainTensor& present_key) {
        dnnl::memory q(q_md, strm.get_engine(), query.data<T>());
        dnnl::memory k(k_md, strm.get_engine(), present_key.data<T>());
        qk_prim.execute(strm, {{DNNL_ARG_SRC, q},
                               {DNNL_ARG_WEIGHTS, k},
                               {DNNL_ARG_DST, attn_score}});
    }

    void exec_kv(dnnl::stream strm, PlainTensor& present_value, PlainTensor& output_emb) {
        dnnl::memory v(v_md, strm.get_engine(), present_value.data<T>());
        dnnl::memory out(out_md, strm.get_engine(), output_emb.data<T>());
        wv_prim.execute(strm, {{DNNL_ARG_SRC, attn_weight}, {DNNL_ARG_WEIGHTS, v}, {DNNL_ARG_DST, out}});
    }

    PlainTensor causal_mask;
    bool select_nfltmax_at_0 = false;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // alibi          [B, H, q_len, kv_len]
    // output_emb    [B, L1, H*S]
    void operator()(dnnl::stream strm,
                    PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    const PlainTensor& alibi_mask,
                    const PlainTensor& attention_mask,
                    PlainTensor& output_emb,
                    bool has_out_transpose,
                    bool auto_causal,
                    float d_scale = 0.0f) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto Hk = present_key.size(1);
        auto kv_len = present_key.size(2);

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);

        prepare_prim(strm, query, present_key, present_value, B, H, Hk, q_len, kv_len, head_size, has_out_transpose);
        exec_qk(strm, query, present_key);

        PlainTensor score;
        score.resize({B, H, q_len, kv_len}, static_cast<float*>(attn_score.get_data_handle()));
        PlainTensor weight;
        weight.resize({B, H, q_len, kv_len}, static_cast<T*>(attn_weight.get_data_handle()));
        // softmax
        parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t m) {
            // apply attention mask & sofmax
            auto ncausal = auto_causal ? (kv_len - q_len + m + 1) : kv_len;
            attn_softmax(&score.at<float>({b, h, m, 0}),
                         &weight.at<T>({b, h, m, 0}),
                         d_scale,
                         alibi_mask ? &alibi_mask.at<float>({b, h, m, 0}, true) : nullptr,
                         attention_mask ? &attention_mask.at<float>({b, h, m, 0}, true) : nullptr,
                         causal_mask ? &causal_mask.at<uint8_t>({b, h, m, 0}, true) : nullptr,
                         select_nfltmax_at_0,
                         ncausal,
                         kv_len,
                         precision_of<T>::value);
        });
        exec_kv(strm, present_value, output_emb);
    }
};

#ifdef OV_CPU_WITH_MLAS
template <>
struct MHAKernel<ScaledDotProductAttention::KT_MLAS, float> {
    size_t m_block_size;
    // buffer to hold qk temp
    std::vector<PlainTensor> qk_buffers;

    MHAKernel() {
        m_block_size = 4;
        select_nfltmax_at_0 = false;
        qk_buffers.resize(parallel_get_max_threads(), PlainTensor(true));
    }

    PlainTensor causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor mask, bool _select_nfltmax_at_0) {
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
    void operator()(dnnl::stream strm,
                    PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    const PlainTensor& alibi_mask,
                    const PlainTensor& attention_mask,
                    PlainTensor& output_emb,
                    bool has_out_transpose,
                    bool auto_causal,
                    float d_scale = 0.0f) {
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

        parallel_for3d(B, H, m_blocks, [&](size_t b, size_t h, size_t m_blk) {
            auto thread_id = parallel_get_thread_num();
            if (thread_id < 0)
                OPENVINO_THROW("The calling thread isn't initialized!");
            auto& qk_buf = qk_buffers[thread_id];

            auto m_start = m_blk * m_block_size;
            auto m_end = std::min(m_start + m_block_size, q_len);
            auto m_cnt = m_end - m_start;

            auto kv_len_cache_align = (((kv_len * sizeof(float)) + 63) / 64 * 64) / sizeof(float);
            qk_buf.resize<float>({m_block_size, kv_len_cache_align});
            const float* q_ptr = &query.at<float>({b, h, m_start, 0});
            const float* k_ptr = &present_key.at<float>({b, h / h_each_group_len, 0, 0});
            const float* v_ptr = &present_value.at<float>({b, h / h_each_group_len, 0, 0});

            float* alibi_ptr = nullptr;
            auto alibi_stride = 0;
            if (alibi_mask) {
                alibi_ptr = &alibi_mask.at<float>({b, h, 0, 0}, true);
                if (alibi_mask.size(2) > 1)
                    alibi_stride = alibi_mask.stride(2);
            }
            float* attn_mask_ptr = nullptr;
            auto attn_mask_stride = 0;
            if (attention_mask) {
                attn_mask_ptr = &attention_mask.at<float>({b, h, 0, 0}, true);
                if (attention_mask.size(2) > 1)
                    attn_mask_stride = attention_mask.stride(2);
            }
            uint8_t* cmask_ptr = nullptr;
            auto cmask_stride = 0;
            if (causal_mask) {
                cmask_ptr = &causal_mask.at<uint8_t>({b, h, 0, 0}, true);
                if (causal_mask.size(2) > 1)
                    cmask_stride = causal_mask.stride(2);
            }

            float* qk = &(qk_buf.at<float>({0, 0}));
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
                attn_softmax(qk + (m - m_start) * qk_m_stride,
                             qk + (m - m_start) * qk_m_stride,
                             d_scale,
                             alibi_ptr + m * alibi_stride,
                             attn_mask_ptr + m * attn_mask_stride,
                             cmask_ptr + m * cmask_stride,
                             select_nfltmax_at_0,
                             ncausal,
                             kv_len,
                             ov::element::f32);
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
                       has_out_transpose ? &output_emb.at<float>({b, m_start, h * head_size}) : &output_emb.at<float>({b, h, m_start}),
                       has_out_transpose ? output_emb.stride(1) : output_emb.stride(2),
                       1);
        });
    }
};
#endif

// 2nd token case : only 1 token in query
struct MHASingleToken {
    PlainTensor m_attn_w;
    PlainTensor m_temp;

    MHASingleToken() : m_attn_w(true), m_temp(true) {}

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // alibi
    // attention_mask [B, 1, q_len, kv_len]
    // output_emb    [B, L1, H, S]
    void operator()(PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    const PlainTensor& alibi_mask,
                    const PlainTensor& attention_mask,
                    PlainTensor& output_emb,
                    const PlainTensor& beams,
                    bool has_out_transpose,
                    bool auto_causal,
                    float d_scale = 0.0f) {
        mha_single_token(query, present_key, present_value, alibi_mask, attention_mask, beams, output_emb,
            m_attn_w, m_temp, has_out_transpose, auto_causal, d_scale);
    }
};

template <ScaledDotProductAttention::KernelTypes KType, typename T>
struct ScaledDotProductAttention::AttentionExecutor : public ScaledDotProductAttention::Executor {
    PlainTensor attn_buf;          // f32[[B|1],[H|1], L1|1, L0+L1]

    MHAKernel<KType, T> kernel;
    MHASingleToken kernel_single_token;

    AttentionExecutor() : attn_buf(true) {}

    void prepare_attn_mask(MemoryPtr attn_input) {
        attn_buf.resize<float>(attn_input->getStaticDims());
        auto p = reinterpret_cast<uint8_t*>(attn_input->getData());
        for (size_t i = 0; i < attn_input->getSize(); i++)
            attn_buf.data<float>()[i] = p[i] ? 0.0f : -FLT_MAX;
    }

    void execute(dnnl::stream strm, const Config& config, const std::vector<MemoryPtr>& inputs, const MemoryPtr output,
                 const MemoryPtr presentk_input, const MemoryPtr presentv_input, const MemoryPtr beam_input) override {
        bool has_out_transpose = config.config.output_BLHxS;
        bool fuse_causal_attn = config.config.fuse_causal_attn;
        bool is_causal = config.config.is_causal;
        bool fuse_concat = config.config.fuse_concat;
        auto input_num = inputs.size();
        PlainTensor present_key, present_value;
        PlainTensor q_input;           // f32[B, H, L1, S]
        PlainTensor k_input;           // f32[B, H|1, L1, S] / [B, H|1, L0+L1, S]
        PlainTensor v_input;           // f32[B, H|1, L1, S] / [B, H|1, L0+L1, S]
        PlainTensor beam_table;        // i32[B, max_kvLen]
        float scale_input = 0.0f;
        size_t B, L1, L0, S;

        q_input.reset(inputs[0]);
        k_input.reset(inputs[1]);
        v_input.reset(inputs[2]);
        present_key.reset(presentk_input);
        present_value.reset(presentv_input);
        if (beam_input)
            beam_table.reset(beam_input);
        PlainTensor attn_mask;
        if (input_num > 3) {
            // attn_mask
            if (inputs[3]->getDesc().getPrecision() == ov::element::u8) {
                // bool->f32
                prepare_attn_mask(inputs[3]);
                attn_mask = attn_buf;
            } else {
                attn_mask.reset(inputs[3]);
            }
            // if has scale, attn_mask must be present
            if (input_num > 4) {
                scale_input = *reinterpret_cast<float*>(inputs[4]->getData());
            }
        }

        // q: [B, H, L1, S]
        const auto & permute_axes = config.config.permute_axes;
        if (!permute_axes.empty()) {
            q_input = q_input.permute(permute_axes);
            k_input = k_input.permute(permute_axes);
            v_input = v_input.permute(permute_axes);
            present_key = present_key.permute(permute_axes);
            present_value = present_value.permute(permute_axes);
        }
        B = q_input.size(0);
        L1 = q_input.size(2);
        S = q_input.size(3);
        L0 = present_key.size(2) - L1;
        auto Hk = k_input.size(1);

        if (fuse_concat) {
            k_input.assert_dims({B, Hk, L1, S});
            v_input.assert_dims({B, Hk, L1, S});
        } else {
            k_input.assert_dims({B, Hk, L0 + L1, S});
            v_input.assert_dims({B, Hk, L0 + L1, S});
        }
        present_key.assert_dims({B, Hk, L0 + L1, S});
        present_value.assert_dims({B, Hk, L0 + L1, S});
        if (beam_table)
            beam_table.assert_dims({B, L0 + L1});

        ov::intel_cpu::PlainTensor output_emb(output);

        bool auto_causal;
        bool use_attn_mask;
        if (fuse_causal_attn) {
            assert(attn_mask);
            attn_mask.assert_dims({B, 1, L1, L0 + L1});
            auto_causal = true;
            use_attn_mask = true;
        } else {
            if (is_causal) {
                auto_causal = true;
                use_attn_mask = false;
            } else {
                // no attn_mask but has scale, there is a 1-d fake attn_mask
                if (input_num > 3 && attn_mask.m_rank > 1) {
                    assert(attn_mask);
                    // spec requires at least 3, but torch sl test does use rank 2
                    if (attn_mask.m_rank == 2)
                        attn_mask = attn_mask.reshape({1, 1, attn_mask.m_dims[0], attn_mask.m_dims[1]});
                    else if (attn_mask.m_rank == 3)
                        attn_mask = attn_mask.reshape({1, attn_mask.m_dims[0], attn_mask.m_dims[1], attn_mask.m_dims[2]});
                    auto_causal = false;
                    use_attn_mask = true;
                } else {
                    auto_causal = false;
                    use_attn_mask = false;
                }
            }
        }

        // second token, or first token with pastkv fusing
        bool use_one_token = L1 == 1 || (fuse_concat && L0 > 0);
        if (!use_one_token) {
            // multi-token version
            kernel(strm, q_input, k_input, v_input, {}, use_attn_mask ? attn_mask : PlainTensor(),
                   output_emb, has_out_transpose, auto_causal, scale_input);
        } else {
            // 1-token version
            // for second token, using a special AVX2/AVX512 float path:
            //  1, in matrix mutiply, using AMX is not efficency because the M dimension of A will alway be 1
            //  2, using float will save the repack cost which typically is required for bf16/int8 opt
            //  3, using dot product can leverage the SIMD while easily adapt to indirect kv cache
            kernel_single_token(q_input, present_key, present_value, {}, use_attn_mask ? attn_mask : PlainTensor(),
                        output_emb, beam_table, has_out_transpose, auto_causal, scale_input);
        }
    }
};

ScaledDotProductAttention::ScaledDotProductAttention(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)), m_tmp_reorder(true) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW("CPU: " + errorMessage);
    }

    const auto node = std::dynamic_pointer_cast<const ov::op::v13::ScaledDotProductAttention>(op);
    if (node) {
        m_config.config.is_causal = node->get_causal();
    } else {
        const auto node = std::dynamic_pointer_cast<const ScaledDotProductAttentionWithKVCache>(op);
        m_config.config = node->get_config();
    }
}

void ScaledDotProductAttention::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    auto rtPrecision = getRuntimePrecision();
    auto orginSDPInputNumber = getOriginalInputsNumber() - (m_config.config.fuse_concat ? 3 : 0);

    NodeConfig config;
    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    config.inConfs.resize(getOriginalInputsNumber());
    config.outConfs.resize(getOriginalOutputsNumber());
    config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        rtPrecision, getInputShapeAtPort(0)));
    config.inConfs[1].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        rtPrecision, getInputShapeAtPort(1)));
    config.inConfs[2].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        rtPrecision, getInputShapeAtPort(2)));
    auto nextPortIdx = 3;
    if (orginSDPInputNumber > 3) {
        // attn_mask
        if (getOriginalInputPrecisionAtPort(nextPortIdx) == ov::element::u8) {
            config.inConfs[nextPortIdx].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                ov::element::u8, getInputShapeAtPort(nextPortIdx)));
        } else {
            config.inConfs[nextPortIdx].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                ov::element::f32, getInputShapeAtPort(nextPortIdx)));
        }
        nextPortIdx++;
    }
    if (orginSDPInputNumber > 4) {
        config.inConfs[nextPortIdx].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            ov::element::f32, getInputShapeAtPort(nextPortIdx)));
    }

    if (m_config.config.fuse_concat) {
        // beam_idx
        config.inConfs[orginSDPInputNumber + 0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            ov::element::i32, getInputShapeAtPort(orginSDPInputNumber + 0)));

        // Since the InputMemory nodes are simple proxy for the state memory as well as the init subgraph memory,
        // it doesn't make sense to set the real KV cache precision, since we don't need any precision conversions
        // provided by the common graph logic. We set precisions equal to the precisions of the state nodes to avoid
        // reorder insertion in between MemoryInputSDPA and SDPA nodes.

        auto past_k_input_mem_precision = getParentEdgeAt(orginSDPInputNumber + 1)->getParent()->getOriginalOutputPrecisionAtPort(0);
        // pastk
        config.inConfs[orginSDPInputNumber + 1].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            past_k_input_mem_precision, getInputShapeAtPort(orginSDPInputNumber + 1)));

        auto past_v_input_mem_precision = getParentEdgeAt(orginSDPInputNumber + 2)->getParent()->getOriginalOutputPrecisionAtPort(0);
        // pastv
        config.inConfs[orginSDPInputNumber + 2].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            past_v_input_mem_precision, getInputShapeAtPort(orginSDPInputNumber + 2)));

        config.outConfs[1].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            past_k_input_mem_precision, getOutputShapeAtPort(1)));
        config.outConfs[1].inPlace(-1);
        config.outConfs[2].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            past_v_input_mem_precision, getOutputShapeAtPort(2)));
        config.outConfs[2].inPlace(-1);
    }

    config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        rtPrecision, getOutputShapeAtPort(0)));

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref_any);
}

void ScaledDotProductAttention::createPrimitive() {
    if (m_config.config.fuse_concat) {
        auto desc = getSelectedPrimitiveDescriptor();
        if (desc == nullptr)
            OPENVINO_THROW("has unidentified preferable primitive descriptor");
    }
    auto rtPrecision = getRuntimePrecision();

    ScaledDotProductAttentionKey key = {rtPrecision};

    auto builder = [&](const ScaledDotProductAttentionKey& key) -> std::shared_ptr<Executor> {
        std::shared_ptr<Executor> executor;
        if (rtPrecision == ov::element::bf16) {
            executor = std::make_shared<AttentionExecutor<KT_ONEDNN, ov::bfloat16>>();
        } else {
    #ifdef OV_CPU_WITH_MLAS
            executor = std::make_shared<AttentionExecutor<KT_MLAS, float>>();
    #else
            executor = std::make_shared<AttentionExecutor<KT_ONEDNN, float>>();
    #endif
        }
        return executor;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);
    m_executor = result.first;
}

void ScaledDotProductAttention::execute(dnnl::stream strm) {
    auto orginSDPInputNumber = getOriginalInputsNumber() - (m_config.config.fuse_concat ? 3 : 0);
    std::vector<MemoryPtr> inputs(orginSDPInputNumber);
    auto output = getChildEdgeAt(0)->getMemoryPtr();
    MemoryPtr presentk_input, presentv_input, beam_input;
    for (size_t i = 0; i < orginSDPInputNumber; i++) {
        inputs[i] = getParentEdgeAt(i)->getMemoryPtr();
    }

    if (m_config.config.fuse_concat) {
        // initialization will be also completed in this func
        gatherConcatPastkv(inputs[1], inputs[2], getParentEdgeAt(orginSDPInputNumber)->getMemoryPtr());

        presentk_input = m_k_state->internal_state_mem();
        presentv_input = m_v_state->internal_state_mem();
        beam_input = m_k_state->hidden_state_mem();
    } else {
        presentk_input = inputs[1];
        presentv_input = inputs[2];
    }
    m_executor->execute(strm, m_config, inputs, output, presentk_input, presentv_input, beam_input);
}

bool ScaledDotProductAttention::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!std::dynamic_pointer_cast<const ov::op::v13::ScaledDotProductAttention>(op) &&
            !std::dynamic_pointer_cast<const ScaledDotProductAttentionWithKVCache>(op)) {
            errorMessage = "Only ScaledDotProductAttention or ScaledDotProductAttentionWithKVCache operation are supported";
            return false;
        }
        // expect shape of q: [B, H, L, S]
        auto inRank = op->get_input_partial_shape(0).size();
        if (inRank != 4u) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(inRank);
            return false;
        }
        int orgSDPAInput = static_cast<int>(op->get_input_size());
        const auto node = std::dynamic_pointer_cast<const ScaledDotProductAttentionWithKVCache>(op);
        if (node) {
            if (node->get_config().fuse_concat) {
                orgSDPAInput -= 3;
            }
        }
        if (orgSDPAInput > 3) {
            inRank = op->get_input_partial_shape(3).size();
            if (inRank > 4u) {
                errorMessage = "Doesn't support 'attention mask' with rank: " + std::to_string(inRank);
                return false;
            }
        }
        // using mha should be better for static shapes
        if (!op->is_dynamic()) {
            errorMessage = "Only run in dynamic mode";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void ScaledDotProductAttention::assignState(const std::shared_ptr<VariableStateKVcache>& state, int idx) {
    auto inputNumber = getOriginalInputsNumber();
    if (inputNumber - 2 == static_cast<size_t>(idx)) {
        m_k_state = state;
    } else if (inputNumber - 1 == static_cast<size_t>(idx)) {
        m_v_state = state;
    } else {
        OPENVINO_THROW(
            "Unexpected idx ", idx , " for a state in a node with type: ", getTypeStr(), " and name ", getName());
    }
}

void ScaledDotProductAttention::gatherConcatPastkv(const MemoryPtr& mem_cur_k, const MemoryPtr& mem_cur_v, const MemoryPtr& mem_beam_idx) {
    PlainTensor cur_k;
    cur_k.reset(mem_cur_k);
    if (!m_config.config.permute_axes.empty())
        cur_k = cur_k.permute(m_config.config.permute_axes);

    updateBeamTable(mem_beam_idx, cur_k.size(2));
    updatePastkv(mem_cur_k, mem_cur_v);
}

// Update beam table using beam_idx. For first token, beam table is like [[0, 0, 0, ...], [1, 1, 1, ...], ...],
//   for second token, beam table is updated using gather(beam_table, beam_idx) then appending [0, 1, 2, ...] to the end for itself.
void ScaledDotProductAttention::updateBeamTable(const MemoryPtr& mem_beam_idx, size_t L1) {
    std::vector<size_t> order = {0, 1, 2, 3};
    if (!m_config.config.permute_axes.empty()) {
        order = m_config.config.permute_axes;
    }
    PlainTensor beam_idx, beam_table_k, beam_table_v;
    auto hidden_state_k = m_k_state->hidden_state_mem();
    auto hidden_state_v = m_v_state->hidden_state_mem();
    beam_idx.reset(mem_beam_idx);

    auto B = beam_idx.size(0);
    auto is_reset = m_k_state->is_reset_state() || m_v_state->is_reset_state();
    auto inputNumber = getOriginalInputsNumber();
    auto&& v_dims = getParentEdgeAt(inputNumber - 1)->getMemory().getStaticDims();
    size_t L0 = v_dims.at(order[2]);
    auto B_state = v_dims.at(order[0]);
    OPENVINO_ASSERT(m_k_state->is_reset_state() == m_v_state->is_reset_state(),
        "KV state must be reset simultaneously, please also reset state for ",
        (m_k_state->is_reset_state() ? m_v_state->get_name() : m_k_state->get_name()));
    OPENVINO_ASSERT(B == B_state, "beam idx batch: ", B, " is not equal to batch of state: ", B_state);
    OPENVINO_ASSERT(B * (L0 + L1) > 0, "B or (L0+L1) is zero, B: ", B, ", L0: ", L0, ", L1: ", L1);
    // resize buffer
    if (B * (L0 + L1) > m_k_state->hidden_state_max_size()) {
        auto mem_desc = std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32, Shape{B, (L0 + L1) * 2});

        auto new_hidden_state_k = std::make_shared<Memory>(getEngine(), mem_desc);
        auto new_hidden_state_v = std::make_shared<Memory>(getEngine(), mem_desc);
        PlainTensor new_beam_table_k, new_beam_table_v;
        new_beam_table_k.reset(new_hidden_state_k);
        new_beam_table_v.reset(new_hidden_state_v);
        if (L0 > 0 && !is_reset) {
            beam_table_k.reset(hidden_state_k);
            beam_table_v.reset(hidden_state_v);
            for (size_t b = 0; b < B; b++) {
                std::memcpy(&new_beam_table_k.at<int32_t>({b}), &beam_table_k.at<int32_t>({b}), sizeof(int32_t) * L0);
                std::memcpy(&new_beam_table_v.at<int32_t>({b}), &beam_table_v.at<int32_t>({b}), sizeof(int32_t) * L0);
            }
        }
        m_k_state->assign_hidden_state(new_hidden_state_k);
        m_v_state->assign_hidden_state(new_hidden_state_v);
        m_k_state->assign_hidden_state_max_size(B * (L0 + L1) * 2);
        m_v_state->assign_hidden_state_max_size(B * (L0 + L1) * 2);
        hidden_state_k = new_hidden_state_k;
        hidden_state_v = new_hidden_state_v;
        beam_table_k = new_beam_table_k;
        beam_table_v = new_beam_table_v;
    }
    std::vector<size_t> new_shape{B, (L0 + L1)};
    auto mem_desc = std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32,
        Shape(new_shape),
        new_shape,
        VectorDims{0, 1},
        0,
        VectorDims{},
        hidden_state_k->getDescWithType<BlockedMemoryDesc>()->getStrides());
    hidden_state_k->redefineDesc(mem_desc);
    hidden_state_v->redefineDesc(mem_desc);

    if (!beam_table_k) {
        beam_table_k.reset(hidden_state_k);
        beam_table_v.reset(hidden_state_v);
    }

    // first token
    if (L0 == 0 || is_reset) {
        for (size_t b = 0; b < B; b++) {
            for (size_t l = 0; l < L0 + L1; l++) {
                beam_table_k.at<int32_t>({b, l}) = b;
                beam_table_v.at<int32_t>({b, l}) = b;
            }
        }
        return;
    }

    // beam order is like [0, 1, 2,...]
    bool no_reorder = true;
    for (size_t i = 0; i < B; i++) {
        if (beam_idx.data<int32_t>()[i] != static_cast<int32_t>(i)) {
            no_reorder = false;
            break;
        }
    }

    // reorder
    if (!no_reorder) {
        m_tmp_reorder.resize<int32_t>({B, L0});
        for (size_t i = 0; i < B; i++) {
            std::memcpy(&m_tmp_reorder.at<int32_t>({i}),
                        &beam_table_k.at<int32_t>({i}),
                        sizeof(int32_t) * L0);
        }
        auto* table = beam_idx.data<int32_t>();
        // beam table is same for both k,v state
        for (size_t i = 0; i < B; i++) {
            std::memcpy(&beam_table_k.at<int32_t>({i}),
                        &m_tmp_reorder.at<int32_t>({static_cast<size_t>(table[i])}),
                        sizeof(int32_t) * L0);
            std::memcpy(&beam_table_v.at<int32_t>({i}),
                        &m_tmp_reorder.at<int32_t>({static_cast<size_t>(table[i])}),
                        sizeof(int32_t) * L0);
        }
    }
    // second token itself
    for (size_t i = 0; i < B; i++) {
        for (size_t j = 0; j < L1; j++) {
            beam_table_k.at<int32_t>({i, L0 + j}) = i;
            beam_table_v.at<int32_t>({i, L0 + j}) = i;
        }
    }
}

// Update pastkv using cur_k, cur_v, simply append cur_k, cur_v to the end of pastkv in the state.
void ScaledDotProductAttention::updatePastkv(const MemoryPtr& mem_cur_k, const MemoryPtr& mem_cur_v) {
    std::vector<size_t> order = {0, 1, 2, 3};
    if (!m_config.config.permute_axes.empty()) {
        order = m_config.config.permute_axes;
    }
    PlainTensor cur_k, past_k;
    PlainTensor cur_v, past_v;
    cur_k.reset(mem_cur_k);
    cur_v.reset(mem_cur_v);
    cur_k = cur_k.permute(order);
    cur_v = cur_v.permute(order);
    auto B = cur_k.size(0);
    auto H = cur_k.size(1);
    auto L1 = cur_k.size(2);
    auto S = cur_k.size(3);
    auto reverse = [&order] (const std::vector<size_t>& cur) {
        std::vector<size_t> result(cur.size());
        for (size_t i = 0; i < cur.size(); i++) {
            result[order[i]] = cur[i];
        }
        return result;
    };
    auto internal_mem_k = m_k_state->internal_state_mem();
    auto internal_mem_v = m_v_state->internal_state_mem();

    auto is_reset = m_k_state->is_reset_state();
    auto inputNumber = getOriginalInputsNumber();
    auto&& v_dims = getParentEdgeAt(inputNumber - 1)->getMemory().getStaticDims();
    size_t L0 = v_dims.at(order[2]);
    auto B_state = v_dims.at(order[0]);
    OPENVINO_ASSERT(B == B_state, "pastkv batch: ", B, " is not equal to batch of state: ", B_state);
    OPENVINO_ASSERT(B * (L0 + L1) > 0, "B or (L0+L1) is zero, B: ", B, ", L0: ", L0, ", L1: ", L1);
    // resize buffer
    if (B * H * (L0 + L1) * S > m_k_state->internal_state_max_size()) {
        auto new_shape = {B, H, (L0 + L1) * 2, S};
        auto mem_desc = std::make_shared<CpuBlockedMemoryDesc>(m_kvcache_precision,
            Shape(reverse(new_shape)),
            new_shape,
            order);

        auto new_internal_mem_k = std::make_shared<Memory>(getEngine(), mem_desc);
        auto new_internal_mem_v = std::make_shared<Memory>(getEngine(), mem_desc);

        PlainTensor new_pastk, new_pastv;
        new_pastk.reset(new_internal_mem_k);
        new_pastv.reset(new_internal_mem_v);
        new_pastk = new_pastk.permute(order);
        new_pastv = new_pastv.permute(order);
        if (L0 > 0 && !is_reset) {
            past_k.reset(internal_mem_k);
            past_v.reset(internal_mem_v);
            past_k = past_k.permute(order);
            past_v = past_v.permute(order);
            attn_memcpy(past_k, past_v, new_pastk, new_pastv);
        }
        internal_mem_k = new_internal_mem_k;
        internal_mem_v = new_internal_mem_v;
        past_k = new_pastk;
        past_v = new_pastv;
        m_k_state->assign_internal_state(new_internal_mem_k);
        m_v_state->assign_internal_state(new_internal_mem_v);
        m_k_state->assign_internal_state_max_size(B * H * (L0 + L1) * 2 * S);
        m_v_state->assign_internal_state_max_size(B * H * (L0 + L1) * 2 * S);
    }
    auto new_shape = {B, H, (L0 + L1), S};
    auto mem_desc = std::make_shared<CpuBlockedMemoryDesc>(m_kvcache_precision,
        Shape(reverse(new_shape)),
        new_shape,
        order,
        0,
        VectorDims{},
        internal_mem_k->getDescWithType<BlockedMemoryDesc>()->getStrides());
    internal_mem_k->redefineDesc(mem_desc);
    internal_mem_v->redefineDesc(mem_desc);

    if (!past_k) {
        past_k.reset(internal_mem_k);
        past_v.reset(internal_mem_v);
        past_k = past_k.permute(order);
        past_v = past_v.permute(order);
    }
    if (L0 > 0 && is_reset) {
        auto inputNumber = getOriginalInputsNumber();
        auto k_mem = getParentEdgeAt(inputNumber - 2)->getMemoryPtr();
        auto v_mem = getParentEdgeAt(inputNumber - 1)->getMemoryPtr();
        auto&& k_shape = k_mem->getShape();
        auto&& v_shape = v_mem->getShape();
        if (!k_shape.hasZeroDims() && !v_shape.hasZeroDims()) {
            PlainTensor init_k, init_v;
            init_k.reset(k_mem);
            init_v.reset(v_mem);
            init_k = init_k.permute(order);
            init_v = init_v.permute(order);
            attn_memcpy(init_k, init_v, past_k, past_v);
        }
    }

    attn_memcpy(cur_k, cur_v, past_k.slice(2, L0, L0 + L1), past_v.slice(2, L0, L0 + L1));
}

ov::element::Type ScaledDotProductAttention::getKVCachePrecision() {
    if (m_kvcache_precision != ov::element::undefined)
        return m_kvcache_precision;
    auto rtPrecision = getRuntimePrecision();
    bool enableKVCacheFP16 = m_config.config.fuse_concat && mayiuse(cpu_isa_t::avx2) && rtPrecision != ov::element::bf16;
    m_kvcache_precision = enableKVCacheFP16 ? ov::element::f16 : rtPrecision;

    return m_kvcache_precision;
}

ov::element::Type ScaledDotProductAttention::getRuntimePrecision() const {
    auto rtPrecision = getOriginalInputPrecisionAtPort(0);
    // only support bf16 and f32
    if (rtPrecision != ov::element::bf16 && rtPrecision != ov::element::f32)
        rtPrecision = ov::element::f32;

    size_t H_idx = 1;
    if (!m_config.config.permute_axes.empty()) {
        H_idx = m_config.config.permute_axes[1];
    }
    const auto& qDims = getInputShapeAtPort(0).getDims();
    const auto& kDims = getInputShapeAtPort(1).getDims();
    // if multi-query, enforce fp32 TODO: support BF16
    if (qDims[H_idx] != kDims[H_idx]) {
        rtPrecision = ov::element::f32;
    }

    return rtPrecision;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
