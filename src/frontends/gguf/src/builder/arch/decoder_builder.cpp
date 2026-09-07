// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Topology of the causal decoder family, in the GGML op vocabulary.
//
// Ground truth for the shape of the graph: llama.cpp src/models/{llama,qwen2,qwen3,phi3,minicpm,
// hunyuan-*,gemma*,gpt-oss,olmoe,qwen35}.cpp and the build_norm / build_qkv / build_attn /
// build_attn_mha / build_ffn expansions in src/llama-graph.cpp, plus the KV-cache cpy_k/get_k
// (SET_ROWS + VIEW) in src/llama-kv-cache.cpp. Op-case values follow
// ggml-decoder.cpp::compute_op_case.
//
// Everything architecture-specific has already been resolved into DecoderConfig, and every
// repeated fragment lives in blocks/, so this file is only the ORDER in which a decoder is
// assembled: inputs -> embeddings -> N x (norm, attention|GDN, norm, FFN|MoE) -> norm -> lm_head.

#include "builder/arch/decoder_builder.hpp"

#include <cmath>

#include "builder/blocks/common.hpp"
#include "builder/blocks/ffn.hpp"
#include "builder/blocks/gated_delta_net.hpp"
#include "openvino/op/parameter.hpp"

namespace ov {
namespace frontend {
namespace gguf {

using ov::element::f32;
using ov::element::i32;
using ov::element::i64;

namespace {
// Dynamic token length, used for model-input Parameters only.
constexpr int64_t D = -1;
}  // namespace

DecoderBuilder::DecoderBuilder(const std::map<std::string, GGUFMetaData>& config,
                               std::unordered_map<std::string, ov::Tensor>& weights,
                               std::unordered_map<std::string, GgufTensorType>& qtypes)
    : m_cfg(config, weights),
      m_emit(weights, qtypes, m_cfg.arch),
      m_kv(blocks::KvCachePlan::build(m_cfg)) {
    auto& graph = *m_emit.graph();
    graph.has_rope = true;
    graph.rope_config = m_cfg.rope_config;
    graph.use_per_op_rope = m_cfg.use_per_op_rope;
    graph.swa_window_size = m_cfg.swa_window_size;
}

void DecoderBuilder::build_inputs() {
    // Names match the cgraph decoder's get_graph_input_ov_name; gguf uses i32 for token/position/
    // index inputs.
    m_emit.add_input("inp_tokens", i32, ps({1, 1, 1, D}));
    m_emit.add_input("inp_pos", i32, ps({1, 1, 1, D}));
    m_emit.add_input("inp_out_ids", i32, ps({1, 1, 1, D}));
    m_emit.add_input("self_kq_mask", f32, ps({1, 1, D, D}));
    // gpt-oss alternates sliding-window / full attention; the windowed mask is a separate
    // input. Only added when the model uses SWA.
    if (m_cfg.has_swa) {
        m_emit.add_input("self_kq_mask_swa", f32, ps({1, 1, D, D}));
    }
    // No beam_idx here. It is a beam-search cache-reorder index for an OpenVINO STATEFUL cache,
    // which ggml has no equivalent of -- so the cgraph decoder does not produce one either, and a
    // builder that declared it would give the two decoders different stateless IO. The pass that
    // creates the state creates it (MakeStateful), because that is where its only consumer, the
    // Gather on the past, is emitted.

    // KV-cache update index (consumed by SET_ROWS; unused in the stateful Concat branch).
    m_emit.add_input("inp_kv_idx", i32, ps({1, 1, 1, D}));
    m_emit.set_tensor_meta("inp_kv_idx", ps({1, 1, 1, T}), i32);

    // token_len_per_seq: number of new tokens per sequence; used by TranslateSession's mask
    // slicing (add_sliced_mask) to build KQ_mask_sliced. An extra (Parameter) input.
    auto p = std::make_shared<ov::op::v0::Parameter>(i64, ps({1}));
    p->set_friendly_name("token_len_per_seq");
    p->output(0).set_names({"token_len_per_seq"});
    m_emit.add_extra_input_node("token_len_per_seq", p);
}

std::string DecoderBuilder::build_embeddings() {
    // GET_ROWS(token_embd.weight, inp_tokens) -> "embd"
    m_emit.add_weight("token_embd.weight");
    m_emit.set_tensor_meta("inp_tokens", ps({1, 1, 1, T}), i32);
    std::string cur = m_emit.add_op("GGML_OP_GET_ROWS",
                                    "embd",
                                    {"token_embd.weight", "inp_tokens"},
                                    ps({1, 1, T, m_cfg.n_embd}),
                                    f32);
    // MiniCPM scales the embeddings by a constant.
    if (m_cfg.embedding_scale != 1.0f) {
        cur = blocks::scale(m_emit, cur, m_cfg.embedding_scale, "embd_scaled");
    }
    // muse-glimmer normalizes the token embeddings with a WEIGHTLESS RMSNorm before layer 0
    // (build_norm with a null weight -> plain ggml_rms_norm, no multiplicative term).
    if (m_cfg.scaleless_embd_norm) {
        cur = m_emit.add_op("GGML_OP_RMS_NORM",
                            "embd_normed",
                            {cur},
                            m_emit.shape_of_tensor(cur),
                            f32,
                            0,
                            {{"eps", m_cfg.rms_eps}});
    }
    return cur;
}

void DecoderBuilder::build_per_layer_embeddings(const std::string& embd) {
    // Gemma4: per-layer token embeddings are built before the layer loop and stored as a
    // 4D tensor "per_layer_embd" of shape [n_layer, T, n_embd_per_layer] (ggml logical order).
    // Inside each layer the VIEW op extracts the il-th slice [1, T, n_embd_per_layer].
    // The projection norm is applied per-slice (over the n_embd_per_layer dim).
    //
    // Topology (mirrors llama.cpp build_inp_per_layer + project_per_layer_inputs):
    //   pe_tok  = GET_ROWS(per_layer_token_embd, inp_tokens)  -> [1, n_layer, T, n_embd_per_layer]
    //             scaled by sqrt(n_embd_per_layer)
    //   pe_proj = MUL_MAT(per_layer_model_proj, cur_embd)     -> [1, n_layer, T, n_embd_per_layer]
    //             scaled by 1/sqrt(n_embd) + RMS_NORM(per_layer_proj_norm)
    //   per_layer_embd = (pe_proj + pe_tok) * 1/sqrt(2)       -> [1, n_layer, T, n_embd_per_layer]
    const int n_layer = m_cfg.n_layer;
    const int pe = m_cfg.n_embd_per_layer;

    // Turn a flat per-token per-layer embedding [1, 1, T, n_layer * pe] into the layer-major
    // [1, n_layer, T, pe] the per-layer VIEW slices from.
    //
    // The data is contiguous as [T, n_layer, pe] (one row per token), so this is a reshape that
    // splits the last axis followed by a transpose of the two middle axes -- a single reshape
    // straight to the target would be wrong for T > 1. Emitted as the same two steps ggml uses
    // (ggml_reshape_3d, then ggml_cont(ggml_permute)), so both reach the shared op cases: RESHAPE
    // case 1 splits the last dim into [.., n_layer, pe] and PERMUTE case 1 is the {0,2,1,3}
    // transpose. Fusing them into one node would need a builder-only case for the transpose. No
    // CONT node: translate_cont's PERMUTE case is a pass-through, since translate_permute already
    // emitted a real Transpose, so ggml's ggml_cont would convert to nothing here.
    auto reshape_to_layer_major = [&](const std::string& flat, const std::string& name) {
        const auto& flat_shape = m_emit.shape_of_tensor(flat);
        const int64_t t = flat_shape[2].is_static() ? flat_shape[2].get_length() : 1;
        auto split = m_emit.add_op("GGML_OP_RESHAPE", name + "_split", {flat}, ps({1, t, n_layer, pe}), f32, 1);
        return m_emit.add_op("GGML_OP_PERMUTE", name, {split}, ps({1, n_layer, t, pe}), f32, 1);
    };

    m_emit.add_weight("per_layer_token_embd.weight");
    m_emit.add_weight("per_layer_model_proj.weight");
    m_emit.add_weight("per_layer_proj_norm.weight");
    const int pe_total = pe * n_layer;

    // Token embedding lookup: [1,1,T, pe_total] -> reshape to [1, n_layer, T, pe]
    auto pe_flat = m_emit.add_op("GGML_OP_GET_ROWS",
                                 "pe_tok_flat",
                                 {"per_layer_token_embd.weight", "inp_tokens"},
                                 ps({1, 1, T, pe_total}),
                                 f32);
    const float pe_scale = std::sqrt(static_cast<float>(pe));
    pe_flat = blocks::scale(m_emit, pe_flat, pe_scale, "pe_tok_flat_scaled");
    auto pe_tok = reshape_to_layer_major(pe_flat, "pe_tok");

    // Model projection: MUL_MAT(per_layer_model_proj, embd) -> [1,1,T, pe_total]
    // per_layer_model_proj is [n_embd, pe_total] -> output [pe_total] per token
    auto proj_flat = m_emit.add_op("GGML_OP_MUL_MAT",
                                   "pe_proj_flat",
                                   {"per_layer_model_proj.weight", embd},
                                   ps({1, 1, T, pe_total}),
                                   f32);
    const float proj_scale = 1.0f / std::sqrt(static_cast<float>(m_cfg.n_embd));
    proj_flat = blocks::scale(m_emit, proj_flat, proj_scale, "pe_proj_flat_scaled");

    // Reshape to [1, n_layer, T, pe] for per-slice RMS_NORM
    auto proj_4d = reshape_to_layer_major(proj_flat, "pe_proj_4d");

    // RMS_NORM + per_layer_proj_norm weight (applied over last dim = pe)
    auto proj_norm = m_emit.add_op("GGML_OP_RMS_NORM",
                                   "pe_proj_rms",
                                   {proj_4d},
                                   ps({1, n_layer, T, pe}),
                                   f32,
                                   0,
                                   {{"eps", m_cfg.rms_eps}});
    m_emit.set_tensor_meta("per_layer_proj_norm.weight", ps({1, 1, 1, pe}), f32);
    auto proj_normed = m_emit.add_op("GGML_OP_MUL",
                                     "pe_proj_normed",
                                     {proj_norm, "per_layer_proj_norm.weight"},
                                     ps({1, n_layer, T, pe}),
                                     f32);

    // Sum token embd + projection, scale by 1/sqrt(2)
    auto pe_sum = m_emit.add_op("GGML_OP_ADD", "pe_sum", {proj_normed, pe_tok}, ps({1, n_layer, T, pe}), f32);
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    m_emit.add_op("GGML_OP_SCALE",
                  "per_layer_embd",
                  {pe_sum},
                  ps({1, n_layer, T, pe}),
                  f32,
                  0,
                  {{"scale", inv_sqrt2}, {"bias", 0.0f}});
}

std::string DecoderBuilder::inject_per_layer_embedding(int il, const std::string& cur, const std::string& inpSA) {
    // Gemma4: each layer takes a slice of the pre-projected per-layer embedding (shape
    // [1,1,T,n_embd_per_layer]), gates it through inp_gate.weight (GELU), multiplies by the
    // per-layer slice, projects back to n_embd, post-norms, then adds residual.
    // per_layer_token_embd (global) is pre-projected before the loop.
    const std::string p = "blk." + std::to_string(il) + ".";
    const int pe = m_cfg.n_embd_per_layer;

    // Slice out this layer's [1,1,T,pe] chunk (dim 1 at index il). The second input is passed
    // purely as a shape reference: per_layer_embd is stored layer-major, so the slice's token axis
    // does not necessarily sit where the layer activation keeps its own, and the ops below combine
    // the two elementwise. See the op_case 104 note in op/view.cpp.
    //
    // It has to be a tensor that still carries every token. At the last layer `cur` has already
    // been filtered down to the output rows by the GET_ROWS in the attention tail, while
    // per_layer_embd always holds all T of them, so using `cur` there would ask op_case 104 to
    // reinterpret T*D values into 1*D. Use the layer's unfiltered input instead and let the
    // GET_ROWS below do the filtering.
    const std::string& pl_shape_ref = (il == m_cfg.n_layer - 1) ? inpSA : cur;
    const std::string pl_slice = p + "per_layer_slice";
    m_emit.add_op("GGML_OP_VIEW",
                  pl_slice,
                  {"per_layer_embd", pl_shape_ref},
                  ps({1, 1, T, pe}),
                  f32,
                  104,  // layer-index slice using the "layer_idx" attribute
                  {{"layer_idx", int64_t(il)}});

    // At the last layer, cur is already filtered to 1 token via inp_out_ids. Mirror llama.cpp
    // gemma4.cpp:347-349: also filter pl_slice so the MUL doesn't broadcast it back to the full
    // sequence length.
    std::string pl_slice_used = pl_slice;
    if (il == m_cfg.n_layer - 1) {
        pl_slice_used = p + "per_layer_slice_sel";
        m_emit.add_op("GGML_OP_GET_ROWS", pl_slice_used, {pl_slice, "inp_out_ids"}, ps({1, 1, T, pe}), f32);
    }

    // gate: cur -> inp_gate.weight -> GELU -> [1,1,T,pe]
    m_emit.add_weight(p + "inp_gate.weight");
    auto gated =
        m_emit.add_op("GGML_OP_MUL_MAT", p + "inp_gate_mm", {p + "inp_gate.weight", cur}, ps({1, 1, T, pe}), f32);
    gated = m_emit.add_op("GGML_UNARY_OP_GELU", p + "inp_gate_gelu", {gated}, ps({1, 1, T, pe}), f32);

    // elementwise multiply by per-layer slice
    auto mul_pe = m_emit.add_op("GGML_OP_MUL", p + "pe_mul", {gated, pl_slice_used}, ps({1, 1, T, pe}), f32);

    // project back to n_embd
    m_emit.add_weight(p + "proj.weight");
    auto pe_proj =
        m_emit.add_op("GGML_OP_MUL_MAT", p + "pe_proj", {p + "proj.weight", mul_pe}, ps({1, 1, T, m_cfg.n_embd}), f32);

    // post-norm + residual add
    pe_proj = blocks::rms_norm(m_emit, pe_proj, p + "post_norm.weight", p + "pe_post_norm", m_cfg.rms_eps);
    return m_emit.add_op("GGML_OP_ADD", p + "pe_out", {cur, pe_proj}, ps({1, 1, T, m_cfg.n_embd}), f32);
}

std::string DecoderBuilder::build_layer(int il, const std::string& layer_in) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::string& inpSA = layer_in;
    std::string cur = layer_in;

    // eps for the post-attention / post-FFN norms; 0 in the config means "reuse rms_eps"
    // (muse-glimmer's post-norms use a tighter 1e-8 than its pre-norms).
    const float post_eps = m_cfg.post_norm_eps > 0.0f ? m_cfg.post_norm_eps : m_cfg.rms_eps;

    // attn_norm (key varies by arch: "attn_norm.weight" for most, "post_attention_norm.weight"
    // for exaone4)
    const std::string attn_norm =
        blocks::rms_norm(m_emit, cur, p + m_cfg.attn_norm_key, p + "attn_norm", m_cfg.rms_eps);

    if (m_cfg.is_qwen35 && m_cfg.is_recurrent_layer(il)) {
        // Hybrid stack: 3 of every 4 layers replace attention with a Gated DeltaNet block.
        // It produces the same thing the attention path does -- the sublayer output before
        // the residual -- so the shared FFN tail below is reused verbatim.
        const std::string gdn_out = blocks::gated_delta_net(m_emit, m_cfg, il, attn_norm, T);
        std::string sa = inpSA;
        std::string ao = gdn_out;
        if (il == m_cfg.n_layer - 1) {
            ao = m_emit.add_op("GGML_OP_GET_ROWS",
                               p + "attn_out_g",
                               {gdn_out, "inp_out_ids"},
                               ps({1, 1, T, m_cfg.n_embd}),
                               f32);
            sa = m_emit.add_op("GGML_OP_GET_ROWS",
                               p + "inpSA_g",
                               {inpSA, "inp_out_ids"},
                               ps({1, 1, T, m_cfg.n_embd}),
                               f32);
        }
        auto ffn_inp_r = m_emit.add_op("GGML_OP_ADD", p + "ffn_inp", {ao, sa}, ps({1, 1, T, m_cfg.n_embd}), f32);
        auto ffn_norm_r = blocks::rms_norm(m_emit, ffn_inp_r, p + m_cfg.ffn_norm_key, p + "ffn_norm", m_cfg.rms_eps);
        // Mirror the non-recurrent FFN dispatch below: Qwen3-Next is MoE for every trunk layer
        // (llama.cpp routes both the GDN and full-attention layers through build_moe_ffn), so a
        // real model has no per-layer dense FFN tensors here and dense_ffn's weight_tensor lookup
        // would assert.
        const bool is_moe_layer_r = m_cfg.is_moe && (il >= m_cfg.n_dense_lead);
        auto down_r = is_moe_layer_r   ? blocks::moe_ffn(m_emit, m_cfg, p, ffn_norm_r, T)
                      : m_cfg.is_geglu ? blocks::geglu_ffn(m_emit, m_cfg, p, ffn_norm_r, T)
                                       : blocks::dense_ffn(m_emit, m_cfg, p, ffn_norm_r, T);
        return m_emit.add_op("GGML_OP_ADD", p + "l_out", {down_r, ffn_inp_r}, ps({1, 1, T, m_cfg.n_embd}), f32);
    }

    std::string attn_out = blocks::attention(m_emit, m_cfg, m_kv, il, attn_norm, T);

    // MiniCPM scales the attention sublayer output before the residual add.
    if (m_cfg.residual_scale != 1.0f) {
        attn_out = blocks::scale(m_emit, attn_out, m_cfg.residual_scale, p + "attn_out_scaled");
    }
    std::string sa = inpSA;
    std::string ao = attn_out;
    if (il == m_cfg.n_layer - 1) {
        ao = m_emit.add_op("GGML_OP_GET_ROWS",
                           p + "attn_out_g",
                           {attn_out, "inp_out_ids"},
                           ps({1, 1, T, m_cfg.n_embd}),
                           f32);
        sa = m_emit.add_op("GGML_OP_GET_ROWS", p + "inpSA_g", {inpSA, "inp_out_ids"}, ps({1, 1, T, m_cfg.n_embd}), f32);
    }
    // Gemma2: post-attention RMSNorm applied to the sublayer output before residual add.
    // Applied after GET_ROWS so the selected-token path matches gemma2.cpp's order.
    if (m_cfg.has_attn_post_norm) {
        ao = blocks::rms_norm(m_emit, ao, p + "post_attention_norm.weight", p + "attn_post_norm", post_eps);
    }

    auto ffn_inp = m_emit.add_op("GGML_OP_ADD", p + "ffn_inp", {ao, sa}, ps({1, 1, T, m_cfg.n_embd}), f32);

    // Pre-FFN/MoE norm. Key varies by arch (ffn_norm_key is resolved in DecoderConfig).
    auto ffn_norm = blocks::rms_norm(m_emit, ffn_inp, p + m_cfg.ffn_norm_key, p + "ffn_norm", m_cfg.rms_eps);

    // Hybrid MoE: lead layers (il < n_dense_lead) are always dense regardless of is_moe.
    const bool is_moe_layer = m_cfg.is_moe && (il >= m_cfg.n_dense_lead);
    std::string down = is_moe_layer     ? blocks::moe_ffn(m_emit, m_cfg, p, ffn_norm, T)
                       : m_cfg.is_geglu ? blocks::geglu_ffn(m_emit, m_cfg, p, ffn_norm, T)
                                        : blocks::dense_ffn(m_emit, m_cfg, p, ffn_norm, T);
    // MiniCPM scales the FFN sublayer output before the residual add.
    if (m_cfg.residual_scale != 1.0f) {
        down = blocks::scale(m_emit, down, m_cfg.residual_scale, p + "ffn_out_scaled");
    }
    // Gemma2: post-FFN RMSNorm applied to the FFN output before residual add.
    if (m_cfg.has_ffn_post_norm) {
        down = blocks::rms_norm(m_emit, down, p + "post_ffw_norm.weight", p + "ffn_post_norm", post_eps);
    }

    cur = m_emit.add_op("GGML_OP_ADD", p + "l_out", {down, ffn_inp}, ps({1, 1, T, m_cfg.n_embd}), f32);

    if (m_cfg.n_embd_per_layer > 0) {
        cur = inject_per_layer_embedding(il, cur, inpSA);
    }

    // Gemma4: per-layer output scale (layer_output_scale.weight is a scalar [1]).
    if (m_emit.has_weight(p + "layer_output_scale.weight")) {
        m_emit.add_named_weight(p + "layer_output_scale.weight");
        cur = m_emit.add_op("GGML_OP_MUL",
                            p + "scaled_out",
                            {cur, p + "layer_output_scale.weight"},
                            m_emit.shape_of_tensor(cur),
                            f32);
    }
    return cur;
}

std::string DecoderBuilder::build_head(const std::string& in) {
    // final norm + lm_head
    std::string cur = blocks::rms_norm(m_emit, in, "output_norm.weight", "result_norm", m_cfg.rms_eps);

    const std::string lm_head_w = m_emit.has_weight("output.weight") ? "output.weight" : "token_embd.weight";
    m_emit.add_weight(lm_head_w);
    const int n_vocab = static_cast<int>(m_emit.weight_tensor(lm_head_w).get_shape()[0]);  // rows = vocab
    auto logits = m_emit.add_op("GGML_OP_MUL_MAT", "result_output", {lm_head_w, cur}, ps({1, 1, T, n_vocab}), f32);
    // MiniCPM scales the logits (1/(n_embd/dim_model_base)).
    if (m_cfg.logit_scale != 1.0f) {
        logits = blocks::scale(m_emit, logits, m_cfg.logit_scale, "result_output_scaled");
    }
    // Gemma2/Gemma3 final logit soft-cap: tanh(logits / cap) * cap.
    if (m_cfg.final_logit_soft_cap != 0.0f) {
        logits = blocks::scale(m_emit, logits, 1.0f / m_cfg.final_logit_soft_cap, "logits_softcap_scaled");
        logits =
            m_emit.add_op("GGML_UNARY_OP_TANH", "logits_softcap_tanh", {logits}, m_emit.shape_of_tensor(logits), f32);
        logits = blocks::scale(m_emit, logits, m_cfg.final_logit_soft_cap, "result_output_softcapped");
    }
    return logits;
}

std::shared_ptr<GgufGraph> DecoderBuilder::build() {
    build_inputs();

    std::string cur = build_embeddings();

    // rope freq factors (llama-3 long context, phi-3): an optional 3rd ROPE input.
    if (m_cfg.has_rope_freqs) {
        m_emit.add_weight("rope_freqs.weight");
    }

    // kq_scale, head size, KV heads and rope config are all computed per-layer inside the loop
    // via the DecoderConfig accessors, so SWA and global layers each get the correct
    // head-size-dependent values (e.g. gemma4 SWA layers use head_size_swa).

    if (m_cfg.n_embd_per_layer > 0) {
        build_per_layer_embeddings(cur);
    }

    for (int il = 0; il < m_cfg.n_layer; ++il) {
        cur = build_layer(il, cur);
    }

    const std::string logits = build_head(cur);

    // logits is the primary output; the cache_k/v_l* outputs (appended per layer above)
    // become Assign sinks via MakeStateful.
    auto graph = m_emit.graph();
    graph->model_output_names.insert(graph->model_output_names.begin(), logits);
    return graph;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
