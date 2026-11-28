// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// GLM5-Next (GLM-5.3-Flash) decoder. Reference: the public GLM-5.3-Flash architecture (zai-org/GLM-5.3-Flash config):
//   embeddings repeated into hc = 4 residual streams (mHC)
//   per layer:  hc_pre -> RMSNorm -> KDA (linear) or nope MLA with k-pool DSA selection -> hc_post
//               hc_pre -> RMSNorm -> dense SwiGLU (lead layers) or sigmoid MoE + shared expert -> hc_post
//   head:       mean over streams -> RMSNorm -> lm_head
// The NextN (MTP) block is not built. Weight matmuls, norms, the short convs, the delta-rule scan and the FFN
// blocks go through the shared GGML-op translators; the stream mixing, the KDA gate algebra, MLA attention and
// the DSA selection mask are emitted as OpenVINO ops directly.

#include "glm5next_builder.hpp"

#include <cmath>
#include <cstdlib>
#include <limits>

#include "builder/api/metadata_store.hpp"
#include "builder/blocks/common.hpp"
#include "builder/blocks/ffn.hpp"
#include "builder/decoder_config.hpp"
#include "builder/decoder_metadata.hpp"
#include "builder/graph_emitter.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/op_types.hpp"

namespace ov::frontend::gguf {
namespace {

using ov::element::f32;
using ov::element::i32;
using ov::element::i64;
using Out = ov::Output<ov::Node>;
constexpr int64_t D = -1;

Out ci64(const std::vector<int64_t>& v) {
    return ov::op::v0::Constant::create(i64, {v.size()}, v);
}
Out cf32(float v) {
    return ov::op::v0::Constant::create(f32, {}, {v});
}
Out reshape(const Out& x, const std::vector<int64_t>& shape) {
    return std::make_shared<ov::op::v1::Reshape>(x, ci64(shape), true);
}
Out transpose(const Out& x, const std::vector<int64_t>& order) {
    return std::make_shared<ov::op::v1::Transpose>(x, ci64(order));
}
Out slice_last(const Out& x, int64_t begin, int64_t end) {
    return std::make_shared<ov::op::v8::Slice>(x, ci64({begin}), ci64({end}), ci64({1}), ci64({-1}));
}
Out to_f32(const Out& x) {
    return x.get_element_type() == f32 ? x : Out(std::make_shared<ov::op::v0::Convert>(x, f32));
}

class Glm5NextBuilder : public ModelBuilder {
public:
    Glm5NextBuilder(const BuildContext& context, const DecoderParameters& parameters)
        : m_meta(context.metadata),
          m_arch(context.arch),
          m_cfg(parameters, context.weights->weights),
          m_emit(context.weights->weights, context.arch, context.weights->translators) {
        m_cfg.configure_graph(m_emit, context.metadata);
        read_hparams();
        // every SwiGLU (routed, shared, dense lead) is clamped; llama.cpp reads one limit per layer
        auto exp = m_meta.get_float_array(m_arch + ".swiglu_clamp_exp");
        auto shexp = m_meta.get_float_array(m_arch + ".swiglu_clamp_shexp");
        if (shexp.empty())
            shexp = exp;
        float limit = 0.0f;
        for (const auto* arr : {&exp, &shexp})
            for (size_t il = 0; il < arr->size() && il < static_cast<size_t>(m_cfg.n_layer); ++il) {
                const auto v = static_cast<float>((*arr)[il]);
                OPENVINO_ASSERT(limit == 0.0f || v == limit,
                                "[GGUF] glm5next: per-layer or per-kind SwiGLU clamp limits are not supported");
                limit = v;
            }
        m_cfg.swiglu_clamp = limit > 1e-6f ? limit : 0.0f;
    }

    std::shared_ptr<GgufGraph> build() override {
        build_inputs();
        m_emit.add_weight("token_embd.weight");
        auto embd = m_emit.add_op("GGML_OP_GET_ROWS", "embd", {"token_embd.weight", "inp_tokens"});
        m_emit.value(embd).get_node_shared_ptr()->get_rt_info()["gguf.token_embedding"] = true;

        // streams h [1, T, hc, E]
        Out h = std::make_shared<ov::op::v0::Tile>(reshape(val(embd), {1, -1, 1, m_cfg.n_embd}), ci64({1, 1, m_hc, 1}));

        for (int il = 0; il < m_cfg.n_layer; ++il) {
            const std::string p = "blk." + std::to_string(il) + ".";
            Mix mix = hc_pre(h, p, "attn");
            auto x = blocks::rms_norm(m_emit, mix.y, p + "attn_norm.weight", p + "attn_norm", m_cfg.rms_eps);
            const auto sub = m_is_mla[il] ? mla(il, x) : kda(il, x);
            h = hc_post(val(sub), h, mix);

            mix = hc_pre(h, p, "ffn");
            x = blocks::rms_norm(m_emit, mix.y, p + "ffn_norm.weight", p + "ffn_norm", m_cfg.rms_eps);
            const auto ffn =
                m_cfg.layer_is_moe(il) ? blocks::moe_ffn(m_emit, m_cfg, p, x) : blocks::dense_ffn(m_emit, m_cfg, p, x);
            h = hc_post(val(ffn), h, mix);
        }

        auto mean = std::make_shared<ov::op::v1::ReduceMean>(h, ci64({2}), false);  // [1, T, E]
        auto cur = reg("hc_head", reshape(mean, {1, 1, -1, m_cfg.n_embd}));
        cur = m_emit.add_op("GGML_OP_GET_ROWS", "hc_head_rows", {cur, "inp_out_ids"});
        cur = blocks::rms_norm(m_emit, cur, "output_norm.weight", "result_norm", m_cfg.rms_eps);
        m_emit.add_weight("output.weight");
        const auto logits = m_emit.add_op("GGML_OP_MUL_MAT", "result_output", {"output.weight", cur});

        auto graph = m_emit.graph();
        graph->model_output_names.insert(graph->model_output_names.begin(), logits);
        return graph;
    }

private:
    struct Mix {
        std::string y;  // [1, 1, T, E] pre-weighted sum of the streams
        Out post;       // [1, T, hc, 1]
        Out comb;       // [1, T, hc(dst), hc(src)]
    };

    void read_hparams() {
        const auto key = [&](const std::string& k) {
            return m_arch + "." + k;
        };
        const auto get_int = [&](const std::string& k) {
            const auto v = m_meta.get_int(key(k));
            OPENVINO_ASSERT(v, "[GGUF] glm5next: missing ", key(k));
            return static_cast<int64_t>(*v);
        };
        const auto get_float = [&](const std::string& k, float dflt) {
            const auto v = m_meta.get_float(key(k));
            return v ? static_cast<float>(*v) : dflt;
        };
        m_hc = get_int("hyper_connection.count");
        OPENVINO_ASSERT(m_hc == 4, "[GGUF] glm5next: mHC with hc != 4 is not supported");
        m_sk_iters = get_int("hyper_connection.sinkhorn_iterations");
        m_hc_eps = get_float("hyper_connection.epsilon", 1e-6f);
        m_n_head = get_int("attention.head_count");
        m_kda_hd = get_int("kda.head_dim");
        m_kda_lb = get_float("kda.gate_lower_bound", -std::numeric_limits<float>::infinity());
        m_d_conv = get_int("ssm.conv_kernel");
        m_q_lora = get_int("attention.q_lora_rank");
        m_kv_lora = get_int("attention.kv_lora_rank");
        m_hd_mla = get_int("attention.key_length_mla");
        m_hd_v_mla = get_int("attention.value_length_mla");
        m_idx_h = get_int("attention.indexer.head_count");
        m_idx_d = get_int("attention.indexer.key_length");
        m_idx_topk = get_int("attention.indexer.top_k");
        m_kpool = get_int("attention.indexer.kpool");
        if (const auto t = m_meta.get_bool(key("attention.indexer.kpool_select_tail")))
            m_tail = *t;
        m_ln_eps = get_float("attention.layer_norm_epsilon", 1e-6f);
        // per-layer attention kind: head_count_kv == 0 marks a KDA layer
        const auto kv = m_meta.get_int_array(key("attention.head_count_kv"));
        m_is_mla.resize(m_cfg.n_layer);
        for (int il = 0; il < m_cfg.n_layer; ++il)
            m_is_mla[il] = static_cast<size_t>(il) < kv.size() ? kv[il] != 0 : true;
    }

    std::string reg(const std::string& name, const Out& v) {
        OPENVINO_ASSERT(!m_emit.graph()->values->count(name), "[GGUF] glm5next: duplicate value ", name);
        (*m_emit.graph()->values)[name] = v;
        return name;
    }
    Out val(const std::string& name) {
        return m_emit.value(name);
    }
    Out weight(const std::string& name) {
        m_emit.add_weight(name);
        return to_f32(m_emit.value(name));
    }
    std::string mm(const std::string& w, const std::string& x, const std::string& name) {
        m_emit.add_weight(w);
        return m_emit.add_op("GGML_OP_MUL_MAT", name, {w, x});
    }

    void build_inputs() {
        m_emit.add_input("inp_tokens", i32, ps({1, 1, 1, D}));
        m_emit.add_input("inp_pos", i32, ps({1, 1, 1, D}));
        m_emit.add_input("inp_out_ids", i32, ps({1, 1, 1, D}));
        m_emit.add_input("self_kq_mask", f32, ps({1, 1, D, D}));
        m_emit.add_input("inp_kv_idx", i32, ps({1, 1, 1, D}));
        auto p = std::make_shared<ov::op::v0::Parameter>(i64, ps({1}));
        p->set_friendly_name("token_len_per_seq");
        p->output(0).set_names({"token_len_per_seq"});
        m_emit.add_extra_input_node("token_len_per_seq", p);
    }

    // ---- mHC ------------------------------------------------------------------------------------------------
    Mix hc_pre(const Out& h, const std::string& p, const std::string& kind) {
        const auto tag = p + "hc_" + kind;
        auto flat = reg(tag + "_flat", reshape(h, {1, 1, -1, m_hc * m_cfg.n_embd}));
        flat = m_emit.add_op("GGML_OP_RMS_NORM", tag + "_flat_norm", {flat}, 0, {{"eps", m_cfg.rms_eps}});
        const auto mixes = mm(p + "hc_" + kind + "_fn.weight", flat, tag + "_mixes");  // [1, 1, T, (2+hc)*hc]
        reg(tag + "_scale", reshape(weight(p + "hc_" + kind + "_scale.weight"), {3}));
        reg(tag + "_base", reshape(weight(p + "hc_" + kind + "_base.weight"), {(2 + m_hc) * m_hc}));
        const auto coef = m_emit.add_op("GGML_OP_HC_MIX",
                                        tag + "_coef",
                                        {mixes, tag + "_scale", tag + "_base"},
                                        0,
                                        {{"n_streams", static_cast<int>(m_hc)},
                                         {"sinkhorn_iterations", static_cast<int>(m_sk_iters)},
                                         {"eps", m_hc_eps}});
        const auto c = reshape(val(coef), {1, -1, (2 + m_hc) * m_hc});  // [1, T, 24]
        auto pre = reshape(slice_last(c, 0, m_hc), {1, -1, m_hc, 1});
        Mix mix;
        mix.post = reshape(slice_last(c, m_hc, 2 * m_hc), {1, -1, m_hc, 1});
        // comb is [src][dst]; transpose to [dst][src] so the post mix is a plain MatMul over src.
        mix.comb = transpose(reshape(slice_last(c, 2 * m_hc, (2 + m_hc) * m_hc), {1, -1, m_hc, m_hc}), {0, 1, 3, 2});
        auto y =
            std::make_shared<ov::op::v1::ReduceSum>(std::make_shared<ov::op::v1::Multiply>(h, pre), ci64({2}), false);
        mix.y = reg(tag + "_pre", reshape(y, {1, 1, -1, m_cfg.n_embd}));
        return mix;
    }

    // out[dst] = y * post[dst] + sum_src comb[src][dst] * h[src]
    Out hc_post(const Out& y, const Out& h, const Mix& mix) {
        auto y4 = reshape(y, {1, -1, 1, m_cfg.n_embd});
        auto mixed = std::make_shared<ov::op::v0::MatMul>(mix.comb, h);  // [1, T, dst, E]
        return std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(y4, mix.post), mixed);
    }

    // ---- KDA (Kimi Delta Attention) ---------------------------------------------------------------------------
    std::string kda(int il, const std::string& x) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const int64_t H = m_n_head, Dh = m_kda_hd, d_inner = H * Dh, K = m_d_conv;
        auto& graph = *m_emit.graph();

        auto conv = [&](const std::string& c) {
            const auto lin = mm(p + "attn_" + c + ".weight", x, p + "kda_" + c);
            const std::string cs = "conv_state_" + c + "_l" + std::to_string(il);
            m_emit.add_input(cs, f32, ps({1, 1, d_inner, K - 1}));
            auto t = m_emit.add_op("GGML_OP_TRANSPOSE", p + "kda_" + c + "_t", {lin});
            auto window =
                m_emit.add_op("GGML_OP_CONCAT", p + "kda_" + c + "_window", {cs, t}, 0, {{"concat_axis", int{0}}});
            auto cs_out = m_emit.add_op("GGML_OP_VIEW",
                                        cs + "_out",
                                        {window},
                                        3,
                                        {{"view_slice", std::vector<int64_t>{3, -(K - 1), K - 1}}});
            graph.model_output_names.push_back(cs_out);
            graph.recurrent_states.emplace_back(cs, cs_out);
            // conv weight [d_inner, 1, K] -> [d_inner, K]
            const auto cw =
                reg(p + "ssm_conv1d_" + c + "_2d", reshape(weight(p + "ssm_conv1d_" + c + ".weight"), {d_inner, K}));
            auto out = m_emit.add_op("GGML_OP_SSM_CONV",
                                     p + "kda_" + c + "_conv",
                                     {window, cw},
                                     0,
                                     {{"batch_major", true}, {"fuse_silu", true}});
            return m_emit.add_op("GGML_OP_RESHAPE",
                                 p + "kda_" + c + "_heads",
                                 {out},
                                 6,
                                 {{"reshape_target", std::vector<int64_t>{0, -1, H, Dh}}, {"special_zero", true}});
        };
        const auto q = conv("q");
        const auto k = conv("k");
        const auto v = conv("v");

        // decay gate: g = lb * sigmoid(-(f_b f_a x + dt_bias) * A), per key dimension; ssm_a holds -exp(A_log)
        auto f = val(mm(p + "ssm_f_b.weight", mm(p + "ssm_f_a.weight", x, p + "kda_f_a"), p + "kda_f_b"));
        Out g1 = std::make_shared<ov::op::v1::Add>(f, reshape(weight(p + "ssm_dt.bias"), {d_inner}));
        g1 = std::make_shared<ov::op::v1::Multiply>(reshape(g1, {1, -1, H, Dh}),
                                                    reshape(weight(p + "ssm_a"), {1, 1, H, 1}));
        Out g;
        if (std::isfinite(m_kda_lb)) {
            g = std::make_shared<ov::op::v1::Multiply>(
                std::make_shared<ov::op::v0::Sigmoid>(std::make_shared<ov::op::v1::Multiply>(g1, cf32(-1.0f))),
                cf32(m_kda_lb));
        } else {
            OPENVINO_THROW("[GGUF] glm5next: KDA without a gate lower bound is not supported");
        }
        const auto gate = reg(p + "kda_gate", g);

        auto beta = mm(p + "ssm_beta.weight", x, p + "kda_beta_lin");
        beta = m_emit.add_op("GGML_UNARY_OP_SIGMOID", p + "kda_beta_sig", {beta});
        beta = m_emit.add_op("GGML_OP_RESHAPE",
                             p + "kda_beta",
                             {beta},
                             6,
                             {{"reshape_target", std::vector<int64_t>{0, -1, H, 1}}, {"special_zero", true}});

        const std::string ss = "ssm_state_l" + std::to_string(il);
        m_emit.add_input(ss, f32, ps({1, H, Dh, Dh}));
        const auto new_state = ss + "_out";
        auto attn = m_emit.add_op("GGML_OP_GATED_DELTA_NET",
                                  p + "kda_scan",
                                  {q, k, v, gate, beta, ss},
                                  0,
                                  {{"gdn_state_slots", int64_t{1}},
                                   {"fuse_qk_l2norm", true},
                                   {"qk_l2_norm_eps", 1e-6f},
                                   {"split_outputs", true}},
                                  {new_state});
        graph.model_output_names.push_back(new_state);
        graph.recurrent_states.emplace_back(ss, new_state);

        // RMSNorm(o) * sigmoid(g_b g_a x), per head
        auto normed = blocks::rms_norm(m_emit, attn, p + "ssm_norm.weight", p + "kda_normed", m_cfg.rms_eps);
        auto g2 = val(mm(p + "ssm_g_b.weight", mm(p + "ssm_g_a.weight", x, p + "kda_g_a"), p + "kda_g_b"));
        // the gate takes the scan output's own [B, T, H, Dh] shape: [1, T, ...] as built, [tokens, 1, ...] after
        // SDPAToPagedAttention (a fixed {1, -1, H, Dh} would broadcast to [T, T, H, Dh] there)
        auto g2_like =
            std::make_shared<ov::op::v1::Reshape>(g2,
                                                  std::make_shared<ov::op::v3::ShapeOf>(val(normed), ov::element::i64),
                                                  false);
        auto gated =
            std::make_shared<ov::op::v1::Multiply>(val(normed), std::make_shared<ov::op::v0::Sigmoid>(g2_like));
        const auto merged = reg(p + "kda_gated", reshape(gated, {1, 1, -1, d_inner}));
        return mm(p + "attn_output.weight", merged, p + "kda_out");
    }

    // ---- nope MLA + k-pool DSA --------------------------------------------------------------------------------
    // Rows write into a per-layer cache Parameter through SET_ROWS, the KV-cache pattern MakeStateful turns into a
    // growing state. The cache has the [1, heads = 1, Tc, width] layout SparseAttention takes: MakeStateful grows it
    // along axis 2 with Concat(past, current), the form PagedSparseAttentionFusion matches.
    // rows_b1t: this call's rows as [B, 1, T, width] (B = 1 except after SDPAToPagedAttention, where B = tokens).
    Out cache_rows_bhsd(int il, const std::string& kind, const Out& rows_b1t, int64_t width) {
        const std::string c = "cache_" + kind + "_l" + std::to_string(il);
        m_emit.add_input(c, f32, ps({1, 1, D, width}));
        const auto r = reg(c + "_rows", rows_b1t);
        auto written = m_emit.add_op("GGML_OP_SET_ROWS", c + "_w", {r, "inp_kv_idx", c});
        m_emit.graph()->model_output_names.push_back(written);
        return val(written);
    }

    // MLA + k-pool DSA through the public sparse-attention ops: QSAPrepare(softmax_gate) builds the pooled block
    // keys and carries them (and the incomplete block's rows) as recurrent state, SparseAttentionIndexer picks the
    // top blocks, SparseAttention attends to them plus the incomplete tail over the latent cache (k = v = latent).
    // CB conversion turns these into PagedQSAIndexer + PagedSparseAttention (FuseQSAIndexer,
    // PagedSparseAttentionFusion).
    // (B, T) of this call, from the positions: [1, 1, 1, T] as built, [B, 1, 1, T] after adapt_to_genai, and
    // [tokens, 1, 1, 1] after SDPAToPagedAttention. The builder's activations fold B * T into one row axis; the
    // attention ops need B and T apart so that the paged fusions see tokens on the batch axis.
    Out batch_tokens() {
        auto s = std::make_shared<ov::op::v3::ShapeOf>(val("inp_pos"), ov::element::i64);
        return std::make_shared<ov::op::v8::Gather>(s, ci64({0, 3}), ci64({0}));
    }
    // rows [.., B * T, w...] -> [B, T, tail...] (or [B, tail0, T, tail1...] with t_second)
    Out to_bt(const Out& rows, const std::vector<int64_t>& tail, bool t_second = false) {
        auto bt = batch_tokens();
        auto b = std::make_shared<ov::op::v8::Gather>(bt, ci64({0}), ci64({0}));
        auto t = std::make_shared<ov::op::v8::Gather>(bt, ci64({1}), ci64({0}));
        ov::OutputVector parts{b};
        if (t_second) {
            parts.push_back(ci64({tail[0]}));
            parts.push_back(t);
            for (size_t i = 1; i < tail.size(); i++)
                parts.push_back(ci64({tail[i]}));
        } else {
            parts.push_back(t);
            for (auto d : tail)
                parts.push_back(ci64({d}));
        }
        return std::make_shared<ov::op::v1::Reshape>(rows, std::make_shared<ov::op::v0::Concat>(parts, 0), false);
    }

    // [H, B*T, L] -> [H, B, T, L]
    Out to_bt_heads(const Out& x, int64_t H, int64_t L) {
        auto bt = batch_tokens();
        return std::make_shared<ov::op::v1::Reshape>(
            x,
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{ci64({H}), bt, ci64({L})}, 0),
            false);
    }

    std::string mla_public_dsa(int il, const std::string& x, const std::string& qr, const Out& q_abs, const Out& C) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const int64_t H = m_n_head, Dh = m_hd_mla, Dv = m_hd_v_mla, L = m_kv_lora;
        const int64_t R = m_kpool, Di = m_idx_d, Hi = m_idx_h;
        OPENVINO_ASSERT(m_tail, "[GGUF] glm5next: DSA without the incomplete tail pool is not supported");
        auto& graph = *m_emit.graph();

        auto iq = to_bt(val(mm(p + "indexer.attn_q_b.weight", qr, p + "indexer_q")), {Hi, Di});  // [B, T, Hi, Di]
        auto q_idx = transpose(iq, {0, 2, 1, 3});                                                // [B, Hi, T, Di]
        auto ik = m_emit.add_op("GGML_OP_NORM",
                                p + "indexer_k_ln",
                                {mm(p + "indexer.attn_k.weight", x, p + "indexer_k_lin")},
                                0,
                                {{"eps", m_ln_eps}});
        Out ikn = std::make_shared<ov::op::v1::Add>(
            std::make_shared<ov::op::v1::Multiply>(val(ik), reshape(weight(p + "indexer.k_norm.weight"), {Di})),
            reshape(weight(p + "indexer.k_norm.bias"), {Di}));
        auto ig = val(mm(p + "indexer_compressor_gate.weight", x, p + "indexer_gate"));
        auto raw = to_bt(std::make_shared<ov::op::v0::Concat>(ov::OutputVector{ikn, ig}, -1), {2 * Di});  // [B, T, 2Di]
        auto wts = to_bt(std::make_shared<ov::op::v1::Multiply>(val(mm(p + "indexer.proj.weight", x, p + "indexer_w")),
                                                                cf32(1.0f / std::sqrt(static_cast<float>(Di * Hi)))),
                         {Hi});  // [B, T, Hi]
        // key_length: tokens in the cache after this call = max position + 1
        auto pos = std::make_shared<ov::op::v0::Convert>(reshape(val("inp_pos"), {-1}), i32);
        auto key_length =
            std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::ReduceMax>(pos, ci64({0}), true),
                                              ov::op::v0::Constant::create(i32, {1}, {1}));
        const std::string sb = "idx_blocks_l" + std::to_string(il), si = "idx_incomplete_l" + std::to_string(il);
        m_emit.add_input(sb, f32, ps({1, D, Di}));
        m_emit.add_input(si, f32, ps({1, R - 1, 2 * Di}));
        auto empty3 = ov::op::v0::Constant::create(f32, {1, 0, 0}, std::vector<float>{});
        auto empty1 = ov::op::v0::Constant::create(f32, {0}, std::vector<float>{});
        auto empty_rope = ov::op::v0::Constant::create(f32, {1, 0}, std::vector<float>{});
        // GGUF ape is ne [Di, R], i.e. an OV [R, Di] tensor; QSAPrepare takes pool_bias [Di, R]
        auto bias = transpose(reshape(weight(p + "indexer_compressor_ape.weight"), {R, Di}), {1, 0});
        auto prep = std::make_shared<ov::op::v17::QSAPrepare>(
            ov::OutputVector{raw, val(sb), key_length, empty3, empty3, empty1, val(si), empty_rope, bias},
            R,
            1e-6f,
            "softmax_gate");
        reg(sb + "_out", prep->output(0));
        reg(si + "_out", prep->output(1));
        for (const auto& st : {sb, si}) {
            graph.model_output_names.push_back(st + "_out");
            graph.recurrent_states.emplace_back(st, st + "_out");
        }

        auto sai = std::make_shared<ov::op::v17::SparseAttentionIndexer>(
            ov::OutputVector{q_idx, prep->output(0), key_length, wts},
            R,
            m_idx_topk,
            1);
        // q_abs [H, B*T, L] -> [B, H, T, L]
        auto q4 = transpose(to_bt_heads(q_abs, H, L), {1, 0, 2, 3});
        const auto& kv = C;  // [B, 1, Tc, L], k = v = latent
        auto scale = cf32(1.0f / std::sqrt(static_cast<float>(Dh)));
        auto att = std::make_shared<ov::op::v17::SparseAttention>(
            ov::OutputVector{q4, kv, kv, sai->output(0), sai->output(1), key_length, reshape(scale, {1})},
            R,
            1,
            true);
        auto o_lat = reshape(transpose(att, {1, 0, 2, 3}), {H, -1, L});  // [B, H, T, L] -> [H, B*T, L]
        auto wv_b = reshape(weight(p + "attn_v_b.weight"), {H, Dv, L});
        auto o = std::make_shared<ov::op::v0::MatMul>(o_lat, wv_b, false, true);  // [H, T, Dv]
        const auto merged = reg(p + "kqv_out", reshape(transpose(o, {1, 0, 2}), {1, 1, -1, H * Dv}));
        return mm(p + "attn_output.weight", merged, p + "attn_out");
    }

    std::string mla(int il, const std::string& x) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const int64_t H = m_n_head, Dh = m_hd_mla, L = m_kv_lora;

        auto qr = blocks::rms_norm(m_emit,
                                   mm(p + "attn_q_a.weight", x, p + "q_a"),
                                   p + "attn_q_a_norm.weight",
                                   p + "q_resid",
                                   m_cfg.rms_eps);
        auto q = transpose(reshape(val(mm(p + "attn_q_b.weight", qr, p + "q")), {-1, H, Dh}), {1, 0, 2});  // [H, T, Dh]
        auto lat = blocks::rms_norm(m_emit,
                                    mm(p + "attn_kv_a_mqa.weight", x, p + "kv_a"),
                                    p + "attn_kv_a_norm.weight",
                                    p + "kv_cmpr",
                                    m_cfg.rms_eps);
        const auto C = cache_rows_bhsd(il, "k", to_bt(val(lat), {1, L}, true), L);  // [B, 1, Tc, L]

        // absorb wk_b: q_abs[h] = q[h] wk_b[h]^T   (wk_b [H, L, Dh])
        auto wk_b = reshape(weight(p + "attn_k_b.weight"), {H, L, Dh});
        auto q_abs = std::make_shared<ov::op::v0::MatMul>(q, wk_b, false, true);  // [H, T, L]
        return mla_public_dsa(il, x, qr, q_abs, C);
    }

    GgufMetadata m_meta;
    std::string m_arch;
    DecoderConfig m_cfg;
    GraphEmitter m_emit;

    int64_t m_hc = 4, m_sk_iters = 20, m_n_head = 0, m_kda_hd = 0, m_d_conv = 4;
    int64_t m_q_lora = 0, m_kv_lora = 0, m_hd_mla = 0, m_hd_v_mla = 0;
    int64_t m_idx_h = 0, m_idx_d = 0, m_idx_topk = 0, m_kpool = 4;
    float m_hc_eps = 1e-6f, m_kda_lb = -5.0f, m_ln_eps = 1e-6f;
    bool m_tail = true;
    std::vector<bool> m_is_mla;
};

DecoderParameters glm5next_parameters(const GgufMetadata& metadata) {
    return resolve_decoder_parameters(metadata, RopeMode::Normal, {});
}

}  // namespace

std::shared_ptr<ModelBuilder> make_glm5next_builder(const BuildContext& context) {
    OPENVINO_ASSERT(context.weights, "[GGUF] glm5next builder requires a weight table");
    return std::make_shared<Glm5NextBuilder>(context, glm5next_parameters(context.metadata));
}

}  // namespace ov::frontend::gguf
