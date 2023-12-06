// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_attn.h"

#include <dnnl_extension_utils.h>
#include <onednn/dnnl.h>

#include <algorithm>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <ie_ngraph_utils.hpp>
#include <string>
#include <shape_inference/shape_inference_internal_dyn.hpp>
#include <vector>

#include "openvino/core/parallel.hpp"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/plain_tensor.hpp"
#include <openvino/op/scaled_dot_product_attention.hpp>
#include "common/arbitrary_order_desc_creator.h"

#ifdef OV_CPU_WITH_MLAS
#    include "mlas/sgemm.hpp"
#endif

#include "utils/plain_tensor.hpp"
#include "kernels/scaled_attn/softmax.hpp"
#include "kernels/scaled_attn/mha_single_token.hpp"
#include "kernels/scaled_attn/attn_memcpy.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::Extensions::Cpu::XARCH;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {
namespace node {

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

    void prepare_prim(dnnl::stream strm, size_t B, size_t H, size_t Hk, size_t q_len, size_t kv_len, size_t S, bool has_out_transpose) {
        auto make_dnnl_dims = [](const std::vector<size_t>& dims) {
            dnnl::memory::dims dnnl_dims(dims.size());
            for (size_t i = 0; i < dims.size(); i++)
                dnnl_dims[i] = static_cast<dnnl::memory::dim>(dims[i]);
            return dnnl_dims;
        };
        auto qkv_dt = precision_of<T>::value == ov::element::f32 ? dt::f32 : dt::bf16;
        dnnl::memory::desc cur_q_md(make_dnnl_dims({B, H, q_len, S}), qkv_dt, tag::abcd);
        dnnl::memory::desc cur_k_md(make_dnnl_dims({B, Hk, kv_len, S}), qkv_dt, tag::abcd);
        if (cur_q_md == q_md && cur_k_md == k_md)
            return;

        q_md = cur_q_md;
        k_md = cur_k_md;
        dnnl::memory::desc attn_md(make_dnnl_dims({B, H, q_len, kv_len}), dt::f32, tag::abcd);
        k_md = k_md.permute_axes({0, 1, 3, 2});
        auto qk_pd = dnnl::matmul::primitive_desc(strm.get_engine(), q_md, k_md, attn_md);
        qk_prim = dnnl::matmul(qk_pd);

        weight_md = dnnl::memory::desc(make_dnnl_dims({B, H, q_len, kv_len}), qkv_dt, tag::abcd);
        v_md = dnnl::memory::desc(make_dnnl_dims({B, Hk, kv_len, S}), qkv_dt, tag::abcd);
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

        prepare_prim(strm, B, H, Hk, q_len, kv_len, head_size, has_out_transpose);
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
    PlainTensor q_input;           // f32[B, H, L1, S]
    PlainTensor k_input;           // f32[B, H|1, L1, S] / [B, H|1, L0+L1, S]
    PlainTensor v_input;           // f32[B, H|1, L1, S] / [B, H|1, L0+L1, S]
    PlainTensor beam_table;        // i32[B, max_kvLen]
    PlainTensor attn_buf;          // f32[[B|1],[H|1], L1|1, L0+L1]
    float scale_input = 0.0f;

    MHAKernel<KType, T> kernel;
    MHASingleToken kernel_single_token;

    size_t B, H, L1, L0, S;

    Config config;
    AttentionExecutor(const Config& _config) : attn_buf(true), config(_config) {}

    void prepare_attn_mask(MemoryPtr attn_input) {
        attn_buf.resize<float>(attn_input->getStaticDims());
        auto p = reinterpret_cast<uint8_t*>(attn_input->getData());
        for (size_t i = 0; i < attn_input->getSize(); i++)
            attn_buf.data<float>()[i] = p[i] ? 0.0f : -FLT_MAX;
    }

    void concat_pastkv(const std::vector<MemoryPtr>& inputs,
                       const std::vector<MemoryPtr>& outputs,
                       const PlainTensor& k_input,
                       const PlainTensor& v_input,
                       PlainTensor& past_k_output,
                       PlainTensor& past_v_output) {
        if (config.config.fuse_concat) {
            k_input.assert_dims({B, 0, L1, S}, true);
            v_input.assert_dims({B, 0, L1, S}, true);
            auto past_k_idx = inputs.size() - 2;
            auto past_k_mem = inputs[past_k_idx + 0];
            L0 = past_k_mem->getStaticDims()[2];
            // [B, H, L0, S]
            past_k_output.reset(outputs[1]);
            past_v_output.reset(outputs[2]);
            attn_memcpy(k_input, v_input, past_k_output.slice(2, L0, L0 + L1), past_v_output.slice(2, L0, L0 + L1));
            if (!config.is_concat_inplaced) {
                PlainTensor past_k_input, past_v_input;
                past_k_input.reset(past_k_mem);
                past_v_input.reset(inputs[past_k_idx + 1]);
                attn_memcpy(past_k_input, past_v_input, past_k_output, past_v_output);
            }
        } else {
            // k,v inputs are already concatenated
            L0 = k_input.size(2) - L1;
            k_input.assert_dims({B, 0, L0 + L1, S}, true);
            v_input.assert_dims({B, 0, L0 + L1, S}, true);
            past_k_output = k_input;
            past_v_output = v_input;
        }
    }

    void execute(dnnl::stream strm, const std::vector<MemoryPtr>& inputs, const std::vector<MemoryPtr>& outputs) override {
        bool has_out_transpose = config.config.output_BLHxS;
        bool fuse_causal_attn = config.config.fuse_causal_attn;
        bool is_causal = config.config.is_causal;
        const bool fuse_concat = config.config.fuse_concat;
        auto input_num = inputs.size() - (fuse_concat ? 2 : 0);

        q_input.reset(inputs[0]);
        k_input.reset(inputs[1]);
        v_input.reset(inputs[2]);
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
        B = q_input.size(0);
        H = q_input.size(1);
        L1 = q_input.size(2);
        S = q_input.size(-1);

        PlainTensor present_key, present_value;
        concat_pastkv(inputs, outputs, k_input, v_input, present_key, present_value);

        ov::intel_cpu::PlainTensor output_emb(outputs[0]);

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

        if (L1 > 1) {
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
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
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
    auto rtPrecision = getOriginalInputPrecisionAtPort(0);
    auto orginSDPInputNumber = getOriginalInputsNumber() - (m_config.config.fuse_concat ? 2 : 0);

    bool enableKVCacheFP16 = m_config.config.fuse_concat && mayiuse(cpu_isa_t::avx2) && rtPrecision != ov::element::bf16;

    auto kvCachePrecision = enableKVCacheFP16 ? ov::element::f16 : rtPrecision;

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
        ArbitraryOrderDescCreator cabdDescCreator({2, 0, 1, 3});

        config.inConfs[orginSDPInputNumber + 0].setMemDesc(cabdDescCreator.createSharedDesc(
            kvCachePrecision, getInputShapeAtPort(orginSDPInputNumber + 0)));
        config.inConfs[orginSDPInputNumber + 1].setMemDesc(cabdDescCreator.createSharedDesc(
            kvCachePrecision, getInputShapeAtPort(orginSDPInputNumber + 1)));

        config.outConfs[1].setMemDesc(cabdDescCreator.createSharedDesc(
            kvCachePrecision, getOutputShapeAtPort(1)));
        config.outConfs[1].inPlace(orginSDPInputNumber + 0);
        config.outConfs[2].setMemDesc(cabdDescCreator.createSharedDesc(
            kvCachePrecision, getOutputShapeAtPort(2)));
        config.outConfs[2].inPlace(orginSDPInputNumber + 1);
    }

    config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        rtPrecision, getOutputShapeAtPort(0)));

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref_any);
    // may fallback to abcd without inplace
    if (m_config.config.fuse_concat) {
        config.inConfs[orginSDPInputNumber + 0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            kvCachePrecision, getInputShapeAtPort(orginSDPInputNumber + 0)));
        config.inConfs[orginSDPInputNumber + 1].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            kvCachePrecision, getInputShapeAtPort(orginSDPInputNumber + 1)));
        config.outConfs[1].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            kvCachePrecision, getOutputShapeAtPort(1)));
        config.outConfs[1].inPlace(-1);
        config.outConfs[2].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            kvCachePrecision, getOutputShapeAtPort(2)));
        config.outConfs[2].inPlace(-1);
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref_any);
    }
}

void ScaledDotProductAttention::createPrimitive() {
    if (m_config.config.fuse_concat) {
        auto desc = getSelectedPrimitiveDescriptor();
        if (desc == nullptr)
            OPENVINO_THROW("has unidentified preferable primitive descriptor");

        m_config.is_concat_inplaced = desc->getConfig().outConfs[1].inPlace() >= 0;
    }
    auto rtPrecision = getOriginalInputPrecisionAtPort(0);

    if (rtPrecision == ov::element::bf16) {
        m_executor = std::make_shared<AttentionExecutor<KT_ONEDNN, ov::bfloat16>>(m_config);
    } else {
        // only support bf16/f32
        rtPrecision = ov::element::f32;
#ifdef OV_CPU_WITH_MLAS
        m_executor = std::make_shared<AttentionExecutor<KT_MLAS, float>>(m_config);
#else
        m_executor = std::make_shared<AttentionExecutor<KT_ONEDNN, float>>(m_config);
#endif
    }
}

void ScaledDotProductAttention::execute(dnnl::stream strm) {
    std::vector<MemoryPtr> inputs(getParentEdges().size()), outputs(getChildEdges().size());
    for (size_t i = 0; i < inputs.size(); i++) {
        inputs[i] = getParentEdgeAt(i)->getMemoryPtr();
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        outputs[i] = getChildEdgeAt(i)->getMemoryPtr();
    }
    m_executor->execute(strm, inputs, outputs);
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
                orgSDPAInput -= 2;
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

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
