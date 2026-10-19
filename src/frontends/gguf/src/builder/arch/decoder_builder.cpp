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

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "builder/blocks/common.hpp"
#include "builder/blocks/ffn.hpp"
#include "builder/blocks/gated_delta_net.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf {

using ov::element::f32;
using ov::element::i32;
using ov::element::i64;

namespace {
// Dynamic token length, used for model-input Parameters only.
constexpr int64_t D = -1;
}  // namespace

DecoderBuilder::DecoderBuilder(const std::map<std::string, GGUFMetaData>& config,
                               std::unordered_map<std::string, ov::Tensor>& weights,
                               std::unordered_map<std::string, GgufTensorType>& qtypes,
                               std::optional<RopeMode> rope,
                               const DecoderOptions& options,
                               const std::unordered_map<std::string, CreatorFunction>* translators)
    : m_cfg(config, weights, rope, options),
      m_emit(weights, qtypes, m_cfg.arch, translators),
      m_kv(blocks::KvCachePlan::build(m_cfg)) {
    m_emit.configure_hadamard(config);
    auto& graph = *m_emit.graph();
    graph.has_rope = true;
    graph.rope_config = m_cfg.rope_config;
    graph.use_per_op_rope = m_cfg.use_per_op_rope;
    graph.swa_window_size = m_cfg.swa_window_size;
    if (m_cfg.arch == "eagle3" || m_cfg.arch == "dflash") {
        const auto& captures = std::get<std::vector<int32_t>>(config.at("draft_target_layers"));
        OPENVINO_ASSERT(!captures.empty(), "[GGUF] draft target_layers must not be empty");
        if (m_cfg.arch == "eagle3") {
            OPENVINO_ASSERT(captures.size() == 3, "[GGUF] Eagle3 requires three target layers");
            graph.model_rt_info["eagle3_mode"] = true;
            graph.model_rt_info["hidden_layers_list"] = captures;
        } else {
            graph.model_rt_info["dflash_mode"] = true;
            ov::AnyMap info;
            std::string layers;
            for (const auto layer : captures)
                layers += (layers.empty() ? "" : ",") + std::to_string(layer - 1);
            info["target_layer_ids"] = layers;
            info["mask_token_id"] = std::to_string(std::get<int>(config.at("draft_mask_token_id")));
            info["candidate_position_offset"] = "0";
            graph.model_rt_info["dflash"] = info;
            if (m_emit.has_weight("markov_w1.weight")) {
                graph.model_rt_info["gguf.dflash_block_size"] = m_cfg.draft_block_size;
                graph.model_rt_info["gguf.dflash_markov"] = true;
                graph.model_rt_info["gguf.dflash_sample_from_anchor"] = m_cfg.draft_sample_from_anchor;
            }
            if (m_emit.has_weight("selector_hidden.weight")) {
                graph.model_rt_info["gguf.dflash_selector_top_k"] = m_cfg.draft_selector_top_k;
                graph.model_rt_info["gguf.dflash_block_size"] = m_cfg.draft_block_size;
            }
            if (m_cfg.lilicorr_hidden > 0) {
                graph.model_rt_info["gguf.lilicorr_top_k"] = m_cfg.lilicorr_top_k;
                graph.model_rt_info["gguf.dflash_block_size"] = m_cfg.draft_block_size;
            }
        }
    }
}

void DecoderBuilder::build_inputs() {
    // Names match the cgraph decoder's get_graph_input_ov_name; gguf uses i32 for token/position/
    // index inputs.
    m_emit.add_input("inp_tokens", i32, ps({1, 1, 1, D}));
    m_emit.add_input("inp_pos", i32, ps({1, 1, 1, D}));
    if (!m_cfg.embedding_model)
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
    if (!m_cfg.embedding_model)
        m_emit.add_input("inp_kv_idx", i32, ps({1, 1, 1, D}));

    // token_len_per_seq: number of new tokens per sequence; used by TranslateSession's mask
    // slicing (add_sliced_mask) to build KQ_mask_sliced. An extra (Parameter) input.
    auto p = std::make_shared<ov::op::v0::Parameter>(i64, ps({1}));
    p->set_friendly_name("token_len_per_seq");
    p->output(0).set_names({"token_len_per_seq"});
    m_emit.add_extra_input_node("token_len_per_seq", p);
}

std::string DecoderBuilder::build_embeddings() {
    if ((m_cfg.arch == "eagle3" || m_cfg.arch == "dflash") && !m_emit.has_weight("token_embd.weight")) {
        auto weight = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{-1, m_cfg.n_embd});
        weight->set_friendly_name("gguf.target_token_embedding");
        weight->output(0).set_names({"gguf.target_token_embedding"});
        m_emit.add_extra_input_node("token_embd.weight", weight);
    } else {
        // GET_ROWS(token_embd.weight, inp_tokens) -> "embd"
        m_emit.add_weight("token_embd.weight");
    }
    std::string cur = m_emit.add_op("GGML_OP_GET_ROWS", "embd", {"token_embd.weight", "inp_tokens"});
    // The tagged node ends the lookup that AdaptToGenAI moves into the embedding model.
    const auto mark_token_embedding = [&](const std::string& name) {
        m_emit.value(name).get_node_shared_ptr()->get_rt_info()["gguf.token_embedding"] = true;
    };
    if (!m_cfg.embedding_scale_tokens_only)
        mark_token_embedding(cur);
    // Gemma and MiniCPM scale the embeddings by a constant.
    if (m_cfg.embedding_scale != 1.0f) {
        cur = blocks::scale(m_emit, cur, m_cfg.embedding_scale, "embd_scaled");
    }
    if (m_cfg.embedding_scale_tokens_only)
        mark_token_embedding(cur);
    // muse-glimmer normalizes the token embeddings with a WEIGHTLESS RMSNorm before layer 0
    // (build_norm with a null weight -> plain ggml_rms_norm, no multiplicative term).
    if (m_cfg.scaleless_embd_norm) {
        cur = m_emit.add_op("GGML_OP_RMS_NORM", "embd_normed", {cur}, 0, {{"eps", m_cfg.rms_eps}});
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
        auto split =
            m_emit.add_op("GGML_OP_RESHAPE",
                          name + "_split",
                          {flat},
                          6,
                          {{"reshape_target", std::vector<int64_t>{0, -1, n_layer, pe}}, {"special_zero", true}});
        return m_emit.add_op("GGML_OP_PERMUTE", name, {split}, 1);
    };

    m_emit.add_weight("per_layer_token_embd.weight");
    m_emit.add_weight("per_layer_model_proj.weight");
    m_emit.add_weight("per_layer_proj_norm.weight");

    // Token embedding lookup: [1,1,T, pe_total] -> reshape to [1, n_layer, T, pe]
    auto pe_flat = m_emit.add_op("GGML_OP_GET_ROWS", "pe_tok_flat", {"per_layer_token_embd.weight", "inp_tokens"});
    const float pe_scale = std::sqrt(static_cast<float>(pe));
    pe_flat = blocks::scale(m_emit, pe_flat, pe_scale, "pe_tok_flat_scaled");
    // Moved to the embedding model as per_layer_inputs in AdaptToGenAI's embedding mode.
    m_emit.value(pe_flat).get_node_shared_ptr()->get_rt_info()["gguf.per_layer_token_embedding"] =
        static_cast<int64_t>(n_layer);
    auto pe_tok = reshape_to_layer_major(pe_flat, "pe_tok");

    // Model projection: MUL_MAT(per_layer_model_proj, embd) -> [1,1,T, pe_total]
    // per_layer_model_proj is [n_embd, pe_total] -> output [pe_total] per token
    auto proj_flat = m_emit.add_op("GGML_OP_MUL_MAT", "pe_proj_flat", {"per_layer_model_proj.weight", embd});
    const float proj_scale = 1.0f / std::sqrt(static_cast<float>(m_cfg.n_embd));
    proj_flat = blocks::scale(m_emit, proj_flat, proj_scale, "pe_proj_flat_scaled");

    // Reshape to [1, n_layer, T, pe] for per-slice RMS_NORM
    auto proj_4d = reshape_to_layer_major(proj_flat, "pe_proj_4d");

    // RMS_NORM + per_layer_proj_norm weight (applied over last dim = pe)
    auto proj_norm = m_emit.add_op("GGML_OP_RMS_NORM", "pe_proj_rms", {proj_4d}, 0, {{"eps", m_cfg.rms_eps}});
    auto proj_normed = m_emit.add_op("GGML_OP_MUL", "pe_proj_normed", {proj_norm, "per_layer_proj_norm.weight"});

    // Sum token embd + projection, scale by 1/sqrt(2)
    auto pe_sum = m_emit.add_op("GGML_OP_ADD", "pe_sum", {proj_normed, pe_tok});
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    m_emit.add_op("GGML_OP_SCALE", "per_layer_embd", {pe_sum}, 0, {{"scale", inv_sqrt2}, {"bias", 0.0f}});
}

std::string DecoderBuilder::inject_per_layer_embedding(int il, const std::string& cur, const std::string& inpSA) {
    // Gemma4: each layer takes a slice of the pre-projected per-layer embedding (shape
    // [1,1,T,n_embd_per_layer]), gates it through inp_gate.weight (GELU), multiplies by the
    // per-layer slice, projects back to n_embd, post-norms, then adds residual.
    // per_layer_token_embd (global) is pre-projected before the loop.
    const std::string p = "blk." + std::to_string(il) + ".";

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
                  104,  // layer-index slice using the "layer_idx" attribute
                  {{"layer_idx", int64_t(il)}});

    // At the last layer, cur is already filtered to 1 token via inp_out_ids. Mirror llama.cpp
    // gemma4.cpp:347-349: also filter pl_slice so the MUL doesn't broadcast it back to the full
    // sequence length.
    std::string pl_slice_used = pl_slice;
    if (selects_output_rows(il)) {
        pl_slice_used = p + "per_layer_slice_sel";
        m_emit.add_op("GGML_OP_GET_ROWS", pl_slice_used, {pl_slice, "inp_out_ids"});
    }

    // gate: cur -> inp_gate.weight -> GELU -> [1,1,T,pe]
    m_emit.add_weight(p + "inp_gate.weight");
    auto gated = m_emit.add_op("GGML_OP_MUL_MAT", p + "inp_gate_mm", {p + "inp_gate.weight", cur});
    gated = m_emit.add_op("GGML_UNARY_OP_GELU", p + "inp_gate_gelu", {gated});

    // elementwise multiply by per-layer slice
    auto mul_pe = m_emit.add_op("GGML_OP_MUL", p + "pe_mul", {gated, pl_slice_used});

    // project back to n_embd
    m_emit.add_weight(p + "proj.weight");
    auto pe_proj = m_emit.add_op("GGML_OP_MUL_MAT", p + "pe_proj", {p + "proj.weight", mul_pe});

    // post-norm + residual add
    pe_proj = blocks::rms_norm(m_emit, pe_proj, p + "post_norm.weight", p + "pe_post_norm", m_cfg.rms_eps);
    return m_emit.add_op("GGML_OP_ADD", p + "pe_out", {cur, pe_proj});
}

std::string DecoderBuilder::build_layer(int il, const std::string& layer_in) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::string& inpSA = layer_in;
    std::string cur = layer_in;

    // eps for the post-attention / post-FFN norms; 0 in the config means "reuse rms_eps"
    // (muse-glimmer's post-norms use a tighter 1e-8 than its pre-norms).
    const auto norm = m_cfg.layer_norm ? blocks::layer_norm : blocks::rms_norm;
    const float post_eps = m_cfg.post_norm_eps > 0.0f ? m_cfg.post_norm_eps : m_cfg.rms_eps;

    const std::string attn_norm =
        m_cfg.post_norm_only ? cur : norm(m_emit, cur, p + m_cfg.attn_norm_key, p + "attn_norm", m_cfg.rms_eps);

    if (m_cfg.is_qwen35 && m_cfg.is_recurrent_layer(il)) {
        // Hybrid stack: 3 of every 4 layers replace attention with a Gated DeltaNet block.
        // It produces the same thing the attention path does -- the sublayer output before
        // the residual -- so the shared FFN tail below is reused verbatim.
        const std::string gdn_out = blocks::gated_delta_net(m_emit, m_cfg, il, attn_norm);
        std::string sa = inpSA;
        std::string ao = gdn_out;
        if (selects_output_rows(il)) {
            ao = m_emit.add_op("GGML_OP_GET_ROWS", p + "attn_out_g", {gdn_out, "inp_out_ids"});
            sa = m_emit.add_op("GGML_OP_GET_ROWS", p + "inpSA_g", {inpSA, "inp_out_ids"});
        }
        auto ffn_inp_r = m_emit.add_op("GGML_OP_ADD", p + "ffn_inp", {ao, sa});
        auto ffn_norm_r = blocks::rms_norm(m_emit, ffn_inp_r, p + m_cfg.ffn_norm_key, p + "ffn_norm", m_cfg.rms_eps);
        // Mirror the non-recurrent FFN dispatch below: Qwen3-Next is MoE for every trunk layer
        // (llama.cpp routes both the GDN and full-attention layers through build_moe_ffn), so a
        // real model has no per-layer dense FFN tensors here and dense_ffn's weight_tensor lookup
        // would assert.
        const bool is_moe_layer_r = m_cfg.layer_is_moe(il);
        auto down_r = is_moe_layer_r   ? blocks::moe_ffn(m_emit, m_cfg, p, ffn_norm_r)
                      : m_cfg.is_geglu ? blocks::geglu_ffn(m_emit, m_cfg, p, ffn_norm_r)
                                       : blocks::dense_ffn(m_emit, m_cfg, p, ffn_norm_r);
        return m_emit.add_op("GGML_OP_ADD", p + "l_out", {down_r, ffn_inp_r});
    }

    std::string attn_out = blocks::attention(m_emit, m_cfg, m_kv, il, attn_norm);

    // MiniCPM scales the attention sublayer output before the residual add.
    if (m_cfg.residual_scale != 1.0f) {
        attn_out = blocks::scale(m_emit, attn_out, m_cfg.residual_scale, p + "attn_out_scaled");
    }
    std::string sa = inpSA;
    std::string ao = attn_out;
    if (selects_output_rows(il)) {
        ao = m_emit.add_op("GGML_OP_GET_ROWS", p + "attn_out_g", {attn_out, "inp_out_ids"});
        sa = m_emit.add_op("GGML_OP_GET_ROWS", p + "inpSA_g", {inpSA, "inp_out_ids"});
    }
    // Gemma2: post-attention RMSNorm applied to the sublayer output before residual add.
    // Applied after GET_ROWS so the selected-token path matches gemma2.cpp's order.
    if (m_cfg.has_attn_post_norm) {
        ao = blocks::rms_norm(m_emit, ao, p + m_cfg.attn_post_norm_key, p + "attn_post_norm", post_eps);
    }

    auto ffn_inp = m_emit.add_op("GGML_OP_ADD", p + "ffn_inp", {ao, sa});

    // Pre-FFN/MoE norm. Key varies by arch (ffn_norm_key is resolved in DecoderConfig).
    auto ffn_norm = m_cfg.parallel_ffn ? attn_norm
                    : m_cfg.post_norm_only
                        ? ffn_inp
                        : norm(m_emit, ffn_inp, p + m_cfg.ffn_norm_key, p + "ffn_norm", m_cfg.rms_eps);

    // Hybrid MoE: lead layers (il < n_dense_lead) are always dense regardless of is_moe.
    const bool is_moe_layer = m_cfg.layer_is_moe(il);
    std::string down = is_moe_layer && m_emit.has_weight(p + "pre_ffw_norm_2.weight")
                           ? blocks::gemma4_moe_ffn(m_emit, m_cfg, p, ffn_inp, ffn_norm)
                       : is_moe_layer   ? blocks::moe_ffn(m_emit, m_cfg, p, ffn_norm)
                       : m_cfg.is_geglu ? blocks::geglu_ffn(m_emit, m_cfg, p, ffn_norm)
                                        : blocks::dense_ffn(m_emit, m_cfg, p, ffn_norm);
    // MiniCPM scales the FFN sublayer output before the residual add.
    if (m_cfg.residual_scale != 1.0f) {
        down = blocks::scale(m_emit, down, m_cfg.residual_scale, p + "ffn_out_scaled");
    }
    // Gemma2: post-FFN RMSNorm applied to the FFN output before residual add.
    if (m_cfg.has_ffn_post_norm) {
        down = blocks::rms_norm(m_emit, down, p + m_cfg.ffn_post_norm_key, p + "ffn_post_norm", post_eps);
    }

    cur = m_emit.add_op("GGML_OP_ADD", p + "l_out", {down, ffn_inp});

    if (m_cfg.n_embd_per_layer > 0) {
        cur = inject_per_layer_embedding(il, cur, inpSA);
    }

    // Gemma4: per-layer output scale (layer_output_scale.weight is a scalar [1]).
    if (m_emit.has_weight(p + "layer_output_scale.weight")) {
        m_emit.add_named_weight(p + "layer_output_scale.weight");
        cur = m_emit.add_op("GGML_OP_MUL", p + "scaled_out", {cur, p + "layer_output_scale.weight"});
    }
    return cur;
}

std::string DecoderBuilder::build_head(const std::string& in) {
    // final norm + lm_head
    const auto norm = m_cfg.layer_norm ? blocks::layer_norm : blocks::rms_norm;
    std::string cur = norm(m_emit, in, "output_norm.weight", "result_norm", m_cfg.rms_eps);
    if (m_cfg.arch == "qwen3tts") {
        const auto hidden = "qwen3tts.hidden_states";
        m_emit.add_op("GGML_OP_SCALE", hidden, {cur}, 0, {{"scale", 1.f}, {"bias", 0.f}});
        m_emit.graph()->model_output_names.push_back(hidden);
    }
    if (m_cfg.embedding_model) {
        using namespace ov::op;
        auto embeddings = m_emit.value(cur);
        ov::Output<ov::Node> output = embeddings;
        auto token_axis = v0::Constant::create(i64, {1}, {2});
        if (m_cfg.pooling_type == 1) {
            output = std::make_shared<v1::ReduceMean>(embeddings, token_axis, true);
        } else if (m_cfg.pooling_type == 2 || m_cfg.pooling_type == 3) {
            ov::Output<ov::Node> index = v0::Constant::create(i64, {1}, {0});
            if (m_cfg.pooling_type == 3) {
                auto shape = std::make_shared<v3::ShapeOf>(embeddings);
                auto axis = v0::Constant::create(i64, {}, {0});
                auto count = std::make_shared<v8::Gather>(shape, token_axis, axis);
                index = std::make_shared<v1::Subtract>(count, v0::Constant::create(i64, {1}, {1}));
            }
            output = std::make_shared<v8::Gather>(embeddings, index, token_axis);
        }
        output = std::make_shared<v0::Squeeze>(output, v0::Constant::create(i64, {2}, {0, 1}));
        output.set_names({"embeddings"});
        (*m_emit.graph()->values)["embeddings"] = output;
        return "embeddings";
    }

    const bool draft = m_cfg.arch == "eagle3" || m_cfg.arch == "dflash";
    const std::string lm_head_w = m_emit.has_weight("output.weight") || draft ? "output.weight" : "token_embd.weight";
    if (draft && !m_emit.has_weight(lm_head_w)) {
        auto weight = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{-1, m_cfg.n_embd});
        weight->set_friendly_name("gguf.target_lm_head");
        weight->output(0).set_names({"gguf.target_lm_head"});
        m_emit.add_extra_input_node(lm_head_w, weight);
    } else {
        m_emit.add_weight(lm_head_w);
    }
    auto logits = m_emit.add_op("GGML_OP_MUL_MAT", "result_output", {lm_head_w, cur});
    // MiniCPM scales the logits (1/(n_embd/dim_model_base)).
    const bool transform_logits = m_cfg.arch != "dflash" || m_emit.has_weight("selector_hidden.weight");
    if (transform_logits && m_cfg.logit_scale != 1.0f) {
        logits = blocks::scale(m_emit, logits, m_cfg.logit_scale, "result_output_scaled");
    }
    // Gemma2/Gemma3 final logit soft-cap: tanh(logits / cap) * cap.
    if (transform_logits && m_cfg.final_logit_soft_cap != 0.0f) {
        logits = blocks::scale(m_emit, logits, 1.0f / m_cfg.final_logit_soft_cap, "logits_softcap_scaled");
        logits = m_emit.add_op("GGML_UNARY_OP_TANH", "logits_softcap_tanh", {logits});
        logits = blocks::scale(m_emit, logits, m_cfg.final_logit_soft_cap, "result_output_softcapped");
    }
    if (m_cfg.arch == "dflash" && m_emit.has_weight("d2t")) {
        using namespace ov::op;
        const auto& mapping = m_emit.weights().at("d2t");
        OPENVINO_ASSERT(mapping.get_element_type() == i64 && mapping.get_shape().size() == 1,
                        "[GGUF] DFlash d2t must be a rank-one i64 tensor");
        auto shape = std::make_shared<v3::ShapeOf>(m_emit.value(logits));
        auto leading = std::make_shared<v8::Slice>(shape,
                                                   v0::Constant::create(i64, {1}, {0}),
                                                   v0::Constant::create(i64, {1}, {3}),
                                                   v0::Constant::create(i64, {1}, {1}));
        auto vocab = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(m_emit.value("token_embd.weight")),
                                                  v0::Constant::create(i64, {1}, {0}),
                                                  v0::Constant::create(i64, {}, {0}));
        auto target =
            std::make_shared<v3::Broadcast>(v0::Constant::create(f32, {}, {-std::numeric_limits<float>::infinity()}),
                                            std::make_shared<v0::Concat>(ov::OutputVector{leading, vocab}, 0));
        auto scattered = std::make_shared<v3::ScatterUpdate>(target,
                                                             std::make_shared<v0::Constant>(mapping),
                                                             m_emit.value(logits),
                                                             v0::Constant::create(i64, {}, {3}));
        m_emit.add_extra_input_node("result_output_mapped", scattered);
        logits = "result_output_mapped";
    }
    if (m_cfg.arch == "dflash" && m_emit.has_weight("markov_w1.weight"))
        logits = build_dflash_markov(logits);
    return logits;
}

std::shared_ptr<GgufGraph> DecoderBuilder::build() {
    build_inputs();
    if (m_cfg.deepstack_layers > 0) {
        OPENVINO_ASSERT(m_cfg.deepstack_layers < m_cfg.n_layer, "[GGUF] DeepStack layers exceed the decoder stack");
        m_emit.add_input("deepstack_embeds", f32, ps({D, m_cfg.deepstack_layers, D, m_cfg.n_embd}));
    }

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

    if (m_cfg.arch == "gemma3n") {
        cur = build_gemma3n(cur);
    } else if (m_cfg.arch == "eagle3") {
        cur = build_eagle3(cur);
    } else if (m_cfg.arch == "dflash") {
        cur = build_dflash(cur);
    } else
        for (int il = 0; il < m_cfg.n_layer; ++il) {
            cur = build_layer(il, cur);
            if (il < m_cfg.deepstack_layers) {
                auto visual = m_emit.add_op("GGML_OP_VIEW",
                                            "blk." + std::to_string(il) + ".deepstack_slice",
                                            {"deepstack_embeds"},
                                            3,
                                            {{"view_slice", std::vector<int64_t>{1, il, 1}}});
                cur = m_emit.add_op("GGML_OP_ADD", "blk." + std::to_string(il) + ".deepstack_out", {cur, visual});
            }
        }

    const std::string logits = build_head(cur);
    m_emit.validate_hadamard();

    // logits is the primary output; the cache_k/v_l* outputs (appended per layer above)
    // become Assign sinks via MakeStateful.
    auto graph = m_emit.graph();
    graph->model_output_names.insert(graph->model_output_names.begin(), logits);
    if (m_cfg.arch == "dflash" && m_emit.has_weight("selector_hidden.weight"))
        build_dflash_selector(logits);
    if (m_cfg.arch == "dflash" && m_cfg.lilicorr_hidden > 0)
        build_lilicorr(logits);
    if (m_cfg.arch == "eagle3" && m_emit.has_weight("d2t")) {
        const auto& mapping = m_emit.weights().at("d2t");
        OPENVINO_ASSERT(mapping.get_element_type() == i64, "[GGUF] Eagle3 d2t must be i64");
        auto constant = std::make_shared<ov::op::v0::Constant>(mapping);
        constant->set_friendly_name("d2t");
        m_emit.add_extra_input_node("d2t", constant);
        graph->model_output_names.push_back("d2t");
    }
    return graph;
}

std::string DecoderBuilder::build_dflash_markov(const std::string& logits) {
    using namespace ov::op;
    using O = ov::Output<ov::Node>;
    const auto scalar = [](int64_t value) {
        return v0::Constant::create(i64, {}, {value});
    };
    const auto axis = [](int64_t value) {
        return v0::Constant::create(i64, {1}, {value});
    };
    const auto weight = [&](const std::string& name) -> O {
        if (name.size() >= 7 && name.compare(name.size() - 7, 7, ".weight") == 0)
            m_emit.add_weight(name);
        else
            m_emit.add_named_weight(name);
        return m_emit.value(name);
    };
    const auto w1 = weight("markov_w1.weight");
    const auto w2 = weight("markov_w2.weight");
    OPENVINO_ASSERT(m_cfg.draft_block_size > 0 && w1.get_partial_shape().rank() == 2 &&
                        w2.get_partial_shape().rank() == 2 && w1.get_partial_shape()[1] == w2.get_partial_shape()[1],
                    "[GGUF] invalid DFlash Markov geometry");
    O base = m_emit.value(logits);
    O hidden = m_emit.value("result_norm");
    O length = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(base), scalar(2), scalar(0));
    const auto column = [&](O value, int64_t index) -> O {
        O safe = std::make_shared<v1::Minimum>(scalar(index), std::make_shared<v1::Subtract>(length, scalar(1)));
        return std::make_shared<v8::Gather>(value, std::make_shared<v0::Unsqueeze>(safe, axis(0)), scalar(2));
    };
    O prev = std::make_shared<v8::Gather>(m_emit.value("inp_tokens"), scalar(0), scalar(3));
    const bool confidence = m_emit.has_weight("conf_proj.weight");
    ov::OutputVector columns, scores;
    for (int64_t index = 0; index < m_cfg.draft_block_size; ++index) {
        O value = column(base, index);
        if (index || m_cfg.draft_sample_from_anchor) {
            O latent = std::make_shared<v8::Gather>(w1, prev, scalar(0));
            O bias = std::make_shared<v0::MatMul>(latent, w2, false, true);
            if (m_emit.has_weight("markov_w2.scale"))
                bias = std::make_shared<v1::Multiply>(bias, weight("markov_w2.scale"));
            if (m_emit.has_weight("d2t")) {
                auto full = std::make_shared<v3::Broadcast>(v0::Constant::create(f32, {}, {0.f}),
                                                            std::make_shared<v3::ShapeOf>(value));
                bias = std::make_shared<v3::ScatterUpdate>(full,
                                                           std::make_shared<v0::Constant>(m_emit.weights().at("d2t")),
                                                           bias,
                                                           scalar(3));
            }
            value = std::make_shared<v1::Add>(value, bias);
            if (confidence) {
                O features = std::make_shared<v0::Concat>(ov::OutputVector{column(hidden, index), latent}, 3);
                O score = std::make_shared<v0::MatMul>(features, weight("conf_proj.weight"), false, true);
                if (m_emit.has_weight("conf_proj.bias"))
                    score = std::make_shared<v1::Add>(score, weight("conf_proj.bias"));
                scores.push_back(std::make_shared<v0::Sigmoid>(score));
            }
            prev =
                std::make_shared<v0::Squeeze>(make_topk_indices(value, scalar(1), -1, v11::TopK::Mode::MAX), axis(3));
        } else if (confidence) {
            scores.push_back(std::make_shared<v0::Sigmoid>(std::make_shared<v8::Gather>(value, axis(0), scalar(3))));
        }
        columns.push_back(value);
    }
    const auto trim = [&](O value) -> O {
        return std::make_shared<v8::Slice>(value,
                                           axis(0),
                                           std::make_shared<v0::Unsqueeze>(length, axis(0)),
                                           axis(1),
                                           axis(2));
    };
    O tail = std::make_shared<v8::Slice>(base,
                                         axis(m_cfg.draft_block_size),
                                         std::make_shared<v0::Unsqueeze>(length, axis(0)),
                                         axis(1),
                                         axis(2));
    O chain = trim(std::make_shared<v0::Concat>(ov::OutputVector{std::make_shared<v0::Concat>(columns, 2), tail}, 2));
    O result = std::make_shared<v1::Select>(std::make_shared<v1::Greater>(length, scalar(m_cfg.draft_block_size)),
                                            base,
                                            chain);
    result.get_node_shared_ptr()->set_friendly_name("dflash.markov_logits");
    m_emit.add_extra_input_node("dflash.markov_logits", result.get_node_shared_ptr());
    if (confidence) {
        O padding = std::make_shared<v8::Gather>(tail, axis(0), scalar(3));
        padding = std::make_shared<v3::Broadcast>(v0::Constant::create(f32, {}, {0.f}),
                                                  std::make_shared<v3::ShapeOf>(padding));
        O score =
            trim(std::make_shared<v0::Concat>(ov::OutputVector{std::make_shared<v0::Concat>(scores, 2), padding}, 2));
        score = std::make_shared<v3::Broadcast>(score, std::make_shared<v3::ShapeOf>(hidden));
        O empty = std::make_shared<v3::Broadcast>(v0::Constant::create(f32, {}, {0.f}),
                                                  std::make_shared<v3::ShapeOf>(hidden));
        score = std::make_shared<v1::Select>(std::make_shared<v1::Greater>(length, scalar(m_cfg.draft_block_size)),
                                             empty,
                                             score);
        score.get_node_shared_ptr()->set_friendly_name("dflash.confidence");
        m_emit.add_extra_input_node("dflash.confidence", score.get_node_shared_ptr());
        m_emit.graph()->model_output_names.push_back("dflash.confidence");
    }
    return "dflash.markov_logits";
}

void DecoderBuilder::build_lilicorr(const std::string& logits) {
    using namespace ov::op;
    using O = ov::Output<ov::Node>;
    const int h = m_cfg.lilicorr_hidden;
    const int k = m_cfg.lilicorr_top_k;
    const int slots = m_cfg.draft_block_size - 1;
    const int nodes = slots * k;
    const auto scalar = [](int64_t x) {
        return v0::Constant::create(i64, {}, {x});
    };
    const auto axis = [](int64_t x) {
        return v0::Constant::create(i64, {1}, {x});
    };
    const auto reshape = [](O x, std::vector<int64_t> shape, bool zeros = false) -> O {
        return std::make_shared<v1::Reshape>(x, v0::Constant::create(i64, {shape.size()}, shape), zeros);
    };
    const auto weight = [&](const std::string& name) -> O {
        m_emit.add_named_weight(name);
        return m_emit.value(name);
    };
    const auto linear = [&](O x, const std::string& name) -> O {
        m_emit.add_weight(name + ".weight");
        O y = std::make_shared<v0::MatMul>(x, m_emit.value(name + ".weight"), false, true);
        if (m_emit.has_weight(name + ".scale"))
            y = std::make_shared<v1::Multiply>(y, weight(name + ".scale"));
        if (m_emit.has_weight(name + ".bias"))
            y = std::make_shared<v1::Add>(y, weight(name + ".bias"));
        return y;
    };
    const auto norm = [&](O x, const std::string& name, float eps, bool center) -> O {
        if (center)
            x = std::make_shared<v1::Subtract>(x, std::make_shared<v1::ReduceMean>(x, axis(-1), true));
        O square = std::make_shared<v1::Multiply>(x, x);
        O variance = std::make_shared<v1::ReduceMean>(square, axis(-1), true);
        O denominator =
            std::make_shared<v0::Sqrt>(std::make_shared<v1::Add>(variance, v0::Constant::create(f32, {}, {eps})));
        O y = std::make_shared<v1::Multiply>(std::make_shared<v1::Divide>(x, denominator), weight(name + ".weight"));
        return m_emit.has_weight(name + ".bias") ? std::make_shared<v1::Add>(y, weight(name + ".bias"))->output(0) : y;
    };
    const auto silu = [](O x) -> O {
        return std::make_shared<v1::Divide>(
            x,
            std::make_shared<v1::Add>(
                v0::Constant::create(f32, {}, {1}),
                std::make_shared<v0::Exp>(std::make_shared<v1::Multiply>(x, v0::Constant::create(f32, {}, {-1})))));
    };
    const auto l2 = [&](O x) -> O {
        auto sum = std::make_shared<v1::ReduceSum>(std::make_shared<v1::Multiply>(x, x), axis(-1), true);
        return std::make_shared<v1::Divide>(
            x,
            std::make_shared<v1::Maximum>(std::make_shared<v0::Sqrt>(sum),
                                          v0::Constant::create(f32, {}, {m_cfg.lilicorr_eps})));
    };
    O values = reshape(m_emit.value(logits), {0, m_cfg.draft_block_size, -1}, true);
    O ids = make_topk_indices(values, scalar(k), -1, v11::TopK::Mode::MAX);
    O probabilities = std::make_shared<v8::Softmax>(values, -1);
    O probs = std::make_shared<v8::Gather>(probabilities, ids, scalar(2), 2);
    auto indices = std::make_shared<v4::Range>(scalar(1), scalar(slots + 1), scalar(1), i64);
    ids = std::make_shared<v8::Gather>(ids, indices, scalar(1));
    probs = std::make_shared<v8::Gather>(probs, indices, scalar(1));
    O log_probs = std::make_shared<v0::Log>(probs);
    O relative = std::make_shared<v1::Subtract>(log_probs, std::make_shared<v8::Gather>(log_probs, axis(0), scalar(2)));
    std::vector<float> ranks(k), first(k, 0.f);
    first[0] = 1.f;
    for (int i = 0; i < k; ++i)
        ranks[i] = k == 1 ? 0.f : float(i) / (k - 1);
    O rank_values = std::make_shared<v3::Broadcast>(v0::Constant::create(f32, {size_t(k)}, ranks),
                                                    std::make_shared<v3::ShapeOf>(probs));
    O first_values = std::make_shared<v3::Broadcast>(v0::Constant::create(f32, {size_t(k)}, first),
                                                     std::make_shared<v3::ShapeOf>(probs));
    ov::OutputVector features;
    for (O x : {log_probs, probs, relative, rank_values, first_values})
        features.push_back(std::make_shared<v0::Unsqueeze>(x, axis(3)));
    O feature_states = reshape(std::make_shared<v0::Concat>(features, 3), {0, nodes, 5}, true);
    const auto feature_layer = "blk." + std::to_string(m_cfg.n_layer + m_cfg.lilicorr_layers) + ".";
    feature_states = norm(feature_states, feature_layer + "ffn_norm", 1e-5f, true);
    feature_states = linear(silu(linear(feature_states, feature_layer + "ffn_up")), feature_layer + "ffn_down");
    O token_states = std::make_shared<v8::Gather>(m_emit.value("token_embd.weight"), ids, scalar(0));
    token_states = reshape(token_states, {0, nodes, m_cfg.n_embd}, true);
    if (h != m_cfg.n_embd)
        token_states = linear(token_states, "lilicorr.token_proj");
    O passes = reshape(m_emit.value("result_norm"), {0, m_cfg.draft_block_size, m_cfg.n_embd}, true);
    passes = linear(std::make_shared<v8::Gather>(passes, indices, scalar(1)), "lilicorr.pass_hidden_proj");
    passes = std::make_shared<v0::Unsqueeze>(passes, axis(2));
    auto pass_shape = std::make_shared<v0::Concat>(
        ov::OutputVector{std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(passes), axis(0), scalar(0)),
                         v0::Constant::create(i64, {3}, {slots, k, h})},
        0);
    passes = reshape(std::make_shared<v3::Broadcast>(passes, pass_shape), {0, nodes, h}, true);
    O positions = std::make_shared<v1::Add>(reshape(weight("lilicorr.slot_embedding.weight"), {1, slots, 1, h}),
                                            reshape(weight("lilicorr.rank_embedding.weight"), {1, 1, k, h}));
    positions = reshape(positions, {1, nodes, h});
    O x = std::make_shared<v1::Add>(std::make_shared<v1::Add>(token_states, passes),
                                    std::make_shared<v1::Add>(feature_states, positions));
    O context = reshape(m_emit.value("dflash.encoder_norm"), {0, -1, m_cfg.n_embd}, true);
    O context_length = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(context), axis(1), scalar(0));
    O anchor_index = std::make_shared<v1::Subtract>(context_length, axis(1));
    O anchor = linear(std::make_shared<v8::Gather>(context, anchor_index, scalar(1)), "lilicorr.context_proj");
    for (int layer = 0; layer < m_cfg.lilicorr_layers; ++layer) {
        const auto prefix = "blk." + std::to_string(m_cfg.n_layer + layer) + ".";
        O input = norm(x, prefix + "attn_norm", m_cfg.rms_eps, false);
        O qkv = linear(input, "lilicorr.layers." + std::to_string(layer) + ".attn.in_proj");
        ov::OutputVector projected;
        for (int side = 0; side < 3; ++side) {
            O part = std::make_shared<v8::Slice>(qkv, axis(side * h), axis((side + 1) * h), axis(1), axis(2));
            part = reshape(part, {0, nodes, m_cfg.lilicorr_heads, h / m_cfg.lilicorr_heads}, true);
            projected.push_back(std::make_shared<v1::Transpose>(part, v0::Constant::create(i64, {4}, {0, 2, 1, 3})));
        }
        O attention = std::make_shared<v13::ScaledDotProductAttention>(projected[0],
                                                                       projected[1],
                                                                       projected[2],
                                                                       weight("lilicorr.attention_bias.bias"),
                                                                       false);
        attention = reshape(std::make_shared<v1::Transpose>(attention, v0::Constant::create(i64, {4}, {0, 2, 1, 3})),
                            {0, nodes, h},
                            true);
        x = std::make_shared<v1::Add>(x, linear(attention, prefix + "attn_output"));
        O mlp = norm(x, prefix + "ffn_norm", m_cfg.rms_eps, false);
        x = std::make_shared<v1::Add>(x, linear(silu(linear(mlp, prefix + "ffn_up")), prefix + "ffn_down"));
    }
    x = norm(x, "lilicorr.output_norm", m_cfg.rms_eps, false);
    anchor = norm(anchor, "lilicorr.anchor_norm", m_cfg.rms_eps, false);
    auto xshape = std::make_shared<v3::ShapeOf>(x);
    O anchor_nodes = std::make_shared<v3::Broadcast>(anchor, xshape);
    O factors = silu(linear(
        std::make_shared<v0::Concat>(ov::OutputVector{x, anchor_nodes, std::make_shared<v1::Multiply>(x, anchor_nodes)},
                                     2),
        "lilicorr.factor_input_proj"));
    O out = reshape(l2(linear(factors, "lilicorr.out_head")), {0, slots, k, m_cfg.lilicorr_factor}, true);
    O in = reshape(l2(linear(factors, "lilicorr.in_head")), {0, slots, k, m_cfg.lilicorr_factor}, true);
    O anchor_out = std::make_shared<v0::Unsqueeze>(l2(linear(anchor, "lilicorr.anchor_out_head")), axis(1));
    O previous = std::make_shared<v8::Slice>(out, axis(0), axis(slots - 1), axis(1), axis(1));
    auto ashape =
        std::make_shared<v0::Concat>(ov::OutputVector{std::make_shared<v8::Gather>(xshape, axis(0), scalar(0)),
                                                      v0::Constant::create(i64, {3}, {1, k, m_cfg.lilicorr_factor})},
                                     0);
    O predecessors =
        std::make_shared<v0::Concat>(ov::OutputVector{std::make_shared<v3::Broadcast>(anchor_out, ashape), previous},
                                     1);
    O scores = std::make_shared<v1::Multiply>(std::make_shared<v0::MatMul>(predecessors, in, false, true),
                                              v0::Constant::create(f32, {}, {m_cfg.lilicorr_scale}));
    std::vector<float> score_mask(size_t(slots * k * k), 1.f);
    std::fill(score_mask.begin() + k, score_mask.begin() + k * k, 0.f);
    scores = std::make_shared<v1::Multiply>(reshape(scores, {0, slots, k * k}, true),
                                            v0::Constant::create(f32, {1, size_t(slots), size_t(k * k)}, score_mask));
    O row = std::make_shared<v0::Concat>(ov::OutputVector{std::make_shared<v0::Convert>(ids, f32), scores}, 2);
    auto packed = std::make_shared<v12::Pad>(row,
                                             v0::Constant::create(i64, {3}, {0, 1, 0}),
                                             v0::Constant::create(i64, {3}, {0, 0, m_cfg.n_embd - k * (k + 1)}),
                                             v0::Constant::create(f32, {}, {0}),
                                             ov::op::PadMode::CONSTANT);
    packed->set_friendly_name("lilicorr.lattice");
    m_emit.add_extra_input_node("lilicorr.lattice", packed);
    m_emit.graph()->model_output_names.push_back("lilicorr.lattice");
}

void DecoderBuilder::build_dflash_selector(const std::string& logits) {
    using namespace ov::op;
    const auto scalar = [](int64_t value) {
        return v0::Constant::create(i64, {}, {value});
    };
    const auto axis = [](int64_t value) {
        return v0::Constant::create(i64, {1}, {value});
    };
    auto raw_values = m_emit.value(logits);
    auto vocab = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(raw_values), scalar(-1), scalar(0));
    auto values = std::make_shared<v1::Reshape>(
        raw_values,
        std::make_shared<v0::Concat>(ov::OutputVector{v0::Constant::create(i64, {3}, {0, 1, -1}),
                                                      std::make_shared<v0::Unsqueeze>(vocab, axis(0))},
                                     0),
        true);
    const auto k = m_cfg.draft_selector_top_k;
    auto candidates = make_topk_indices(values, scalar(k), -1, v11::TopK::Mode::MAX);
    if (k > 1) {
        std::vector<int64_t> order(k);
        std::iota(order.begin(), order.end(), int64_t{0});
        std::swap(order[0], order[1]);
        candidates = std::make_shared<v8::Gather>(candidates, v0::Constant::create(i64, {size_t(k)}, order), scalar(3));
    }
    auto unary = std::make_shared<v8::Gather>(values, candidates, scalar(3), 3);
    auto length = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(values), scalar(2), scalar(0));
    auto end = std::make_shared<v1::Minimum>(length, scalar(m_cfg.draft_block_size));
    auto slots = std::make_shared<v4::Range>(scalar(0), end, scalar(1), i64);
    auto previous_slots = std::make_shared<v1::Maximum>(std::make_shared<v1::Subtract>(slots, scalar(1)), scalar(0));
    auto current_ids = std::make_shared<v8::Gather>(candidates, slots, scalar(2));
    auto previous_ids = std::make_shared<v8::Gather>(candidates, previous_slots, scalar(2));
    auto anchor = std::make_shared<v8::Gather>(m_emit.value("inp_tokens"), axis(0), scalar(3));
    auto anchor_ids = std::make_shared<v1::Reshape>(anchor, v0::Constant::create(i64, {4}, {0, 0, 1, 1}), true);
    auto first = std::make_shared<v1::Reshape>(std::make_shared<v1::Equal>(slots, scalar(1)),
                                               v0::Constant::create(i64, {4}, {1, 1, -1, 1}),
                                               false);
    auto predecessors = std::make_shared<v1::Select>(first, anchor_ids, previous_ids);
    for (const auto name : {"selector_hidden.weight", "selector_predecessor.weight", "selector_successor.weight"})
        m_emit.add_weight(name);
    auto gate =
        std::make_shared<v0::MatMul>(m_emit.value("result_norm"), m_emit.value("selector_hidden.weight"), false, true);
    auto gate_run = std::make_shared<v0::Unsqueeze>(std::make_shared<v8::Gather>(gate, slots, scalar(2)), axis(3));
    auto predecessor =
        std::make_shared<v8::Gather>(m_emit.value("selector_predecessor.weight"), predecessors, scalar(0));
    auto successor = std::make_shared<v8::Gather>(m_emit.value("selector_successor.weight"), current_ids, scalar(0));
    auto scores =
        std::make_shared<v0::MatMul>(std::make_shared<v1::Multiply>(predecessor, gate_run), successor, false, true);
    auto unary_run = std::make_shared<v0::Unsqueeze>(std::make_shared<v8::Gather>(unary, slots, scalar(2)), axis(3));
    auto flattened = std::make_shared<v1::Reshape>(std::make_shared<v1::Add>(scores, unary_run),
                                                   v0::Constant::create(i64, {4}, {0, 0, 0, k * k}),
                                                   true);
    auto rows =
        std::make_shared<v0::Concat>(ov::OutputVector{std::make_shared<v0::Convert>(current_ids, f32), flattened}, 3);
    auto live = std::make_shared<v1::Reshape>(std::make_shared<v1::Equal>(slots, scalar(0)),
                                              v0::Constant::create(i64, {4}, {1, 1, -1, 1}),
                                              false);
    auto zero_anchor = std::make_shared<v1::Select>(live, v0::Constant::create(f32, {}, {0}), rows);
    auto packed = std::make_shared<v12::Pad>(zero_anchor,
                                             v0::Constant::create(i64, {4}, {0, 0, 0, 0}),
                                             v0::Constant::create(i64, {4}, {0, 0, 0, m_cfg.n_embd - k * (k + 1)}),
                                             v0::Constant::create(f32, {}, {0}),
                                             ov::op::PadMode::CONSTANT);
    packed->set_friendly_name("dflash.lattice");
    m_emit.add_extra_input_node("dflash.lattice", packed);
    m_emit.graph()->model_output_names.push_back("dflash.lattice");
}

std::string DecoderBuilder::build_dflash_conv(const std::string& hidden,
                                              const std::string& dynamic,
                                              const std::string& base,
                                              int side,
                                              const std::string& name) {
    using namespace ov::op;
    const int groups = m_cfg.n_embd / m_cfg.draft_conv_group;
    const int kernel = m_cfg.draft_conv_kernel;
    m_emit.add_named_weight(base);
    auto coefficients = std::make_shared<v1::Reshape>(m_emit.value(dynamic),
                                                      v0::Constant::create(i64, {7}, {0, 0, 0, 2, kernel, groups, 1}),
                                                      true);
    auto bases =
        std::make_shared<v1::Reshape>(m_emit.value(base),
                                      v0::Constant::create(i64, {4}, {2, kernel, groups, m_cfg.draft_conv_group}),
                                      false);
    auto shape = std::make_shared<v3::ShapeOf>(m_emit.value(hidden));
    auto length =
        std::make_shared<v8::Gather>(shape, v0::Constant::create(i64, {1}, {2}), v0::Constant::create(i64, {}, {0}));
    ov::Output<ov::Node> result;
    for (int tap = 0; tap < kernel; ++tap) {
        auto coefficient = std::make_shared<v8::Gather>(coefficients,
                                                        v0::Constant::create(i64, {}, {side}),
                                                        v0::Constant::create(i64, {}, {3}));
        coefficient = std::make_shared<v8::Gather>(coefficient,
                                                   v0::Constant::create(i64, {}, {tap}),
                                                   v0::Constant::create(i64, {}, {3}));
        auto weight = std::make_shared<v8::Gather>(bases,
                                                   v0::Constant::create(i64, {}, {side}),
                                                   v0::Constant::create(i64, {}, {0}));
        weight = std::make_shared<v8::Gather>(weight,
                                              v0::Constant::create(i64, {}, {tap}),
                                              v0::Constant::create(i64, {}, {0}));
        auto combined = std::make_shared<v1::Reshape>(std::make_shared<v1::Add>(coefficient, weight), shape, false);
        auto padded = std::make_shared<v12::Pad>(m_emit.value(hidden),
                                                 v0::Constant::create(i64, {4}, {0, 0, tap, 0}),
                                                 v0::Constant::create(i64, {4}, {0, 0, 0, 0}),
                                                 v0::Constant::create(f32, {}, {0}),
                                                 ov::op::PadMode::CONSTANT);
        auto shifted = std::make_shared<v8::Slice>(padded,
                                                   v0::Constant::create(i64, {1}, {0}),
                                                   length,
                                                   v0::Constant::create(i64, {1}, {1}),
                                                   v0::Constant::create(i64, {1}, {2}));
        auto term = std::make_shared<v1::Multiply>(combined, shifted);
        result = result.get_node() ? std::make_shared<v1::Add>(result, term)->output(0) : term->output(0);
    }
    m_emit.add_extra_input_node(name, result.get_node_shared_ptr());
    return name;
}

std::string DecoderBuilder::build_dflash(const std::string& embeddings) {
    const auto width = m_emit.weights().at("fc.weight").get_shape().back();
    m_emit.add_input("hidden_states", f32, {1, 1, -1, static_cast<int64_t>(width)});
    m_emit.add_weight("fc.weight");
    auto context = m_emit.add_op("GGML_OP_MUL_MAT", "dflash.fc", {"fc.weight", "hidden_states"});
    if (m_emit.has_weight("fc.scale")) {
        m_emit.add_named_weight("fc.scale");
        context = m_emit.add_op("GGML_OP_MUL", "dflash.fc_scaled", {context, "fc.scale"});
    }
    context = blocks::rms_norm(m_emit, context, "enc.output_norm.weight", "dflash.encoder_norm", m_cfg.rms_eps);
    std::string current = embeddings;
    const auto dynamic_slice = [&](const std::string& value, const std::string& name, bool keep_noise) {
        using namespace ov::op;
        auto context_shape = std::make_shared<v3::ShapeOf>(m_emit.value(context));
        auto noise_shape = std::make_shared<v3::ShapeOf>(m_emit.value(embeddings));
        auto axis = v0::Constant::create(i64, {1}, {2});
        auto scalar_axis = v0::Constant::create(i64, {}, {0});
        auto n_context = std::make_shared<v8::Gather>(context_shape, axis, scalar_axis);
        auto n_noise = std::make_shared<v8::Gather>(noise_shape, axis, scalar_axis);
        ov::Output<ov::Node> begin = keep_noise ? n_context->output(0) : v0::Constant::create(i64, {1}, {0});
        ov::Output<ov::Node> end =
            keep_noise ? std::make_shared<v1::Add>(n_context, n_noise)->output(0) : n_context->output(0);
        auto sliced =
            std::make_shared<v8::Slice>(m_emit.value(value), begin, end, v0::Constant::create(i64, {1}, {1}), axis);
        m_emit.add_extra_input_node(name, sliced);
        return name;
    };
    for (int layer = 0; layer < m_cfg.n_layer; ++layer) {
        const auto p = "blk." + std::to_string(layer) + ".";
        auto normalized = blocks::rms_norm(m_emit, current, p + "attn_norm.weight", p + "noise_norm", m_cfg.rms_eps);
        std::string attn_dynamic;
        if (m_cfg.draft_conv_kernel > 0) {
            m_emit.add_weight(p + "attn_conv_proj.weight");
            attn_dynamic =
                m_emit.add_op("GGML_OP_MUL_MAT", p + "attn_dynamic", {p + "attn_conv_proj.weight", normalized});
            normalized = build_dflash_conv(normalized, attn_dynamic, p + "attn_conv_base", 0, p + "attn_conv_in");
        }
        auto joined = m_emit.add_op("GGML_OP_CONCAT", p + "context", {context, normalized}, 0, {{"concat_axis", 1}});
        auto attended = blocks::attention(m_emit, m_cfg, m_kv, layer, joined);
        attended = dynamic_slice(attended, p + "noise_attention", true);
        if (m_cfg.attention_value_scale != 0.f)
            attended = blocks::scale(m_emit, attended, m_cfg.attention_value_scale, p + "attention_scaled");
        if (!attn_dynamic.empty())
            attended = build_dflash_conv(attended, attn_dynamic, p + "attn_conv_base", 1, p + "attn_conv_out");
        if (m_cfg.has_attn_post_norm)
            attended = blocks::rms_norm(m_emit,
                                        attended,
                                        p + m_cfg.attn_post_norm_key,
                                        p + "post_attention_norm",
                                        m_cfg.rms_eps);
        current = m_emit.add_op("GGML_OP_ADD", p + "attn_residual", {current, attended});
        normalized = blocks::rms_norm(m_emit, current, p + "ffn_norm.weight", p + "ffn_norm", m_cfg.rms_eps);
        std::string ffn_dynamic;
        if (m_cfg.draft_conv_kernel > 0) {
            m_emit.add_weight(p + "ffn_conv_proj.weight");
            ffn_dynamic = m_emit.add_op("GGML_OP_MUL_MAT", p + "ffn_dynamic", {p + "ffn_conv_proj.weight", normalized});
            normalized = build_dflash_conv(normalized, ffn_dynamic, p + "ffn_conv_base", 0, p + "ffn_conv_in");
        }
        auto ffn = m_cfg.is_geglu ? blocks::geglu_ffn(m_emit, m_cfg, p, normalized)
                                  : blocks::dense_ffn(m_emit, m_cfg, p, normalized);
        if (!ffn_dynamic.empty())
            ffn = build_dflash_conv(ffn, ffn_dynamic, p + "ffn_conv_base", 1, p + "ffn_conv_out");
        if (m_cfg.has_ffn_post_norm)
            ffn = blocks::rms_norm(m_emit, ffn, p + m_cfg.ffn_post_norm_key, p + "post_ffw_norm", m_cfg.rms_eps);
        current = m_emit.add_op("GGML_OP_ADD", p + "ffn_residual", {current, ffn});
        if (m_emit.has_weight(p + "layer_output_scale.weight")) {
            m_emit.add_weight(p + "layer_output_scale.weight");
            current = m_emit.add_op("GGML_OP_MUL", p + "scaled_out", {current, p + "layer_output_scale.weight"});
        }
        for (const auto side : {"k", "v"}) {
            const auto cache = "cache_" + std::string(side) + "_l" + std::to_string(layer);
            m_emit.value(cache).get_node_shared_ptr()->get_rt_info()["gguf.draft_context"] =
                m_emit.value(context).get_node_shared_ptr()->get_friendly_name();
        }
    }
    return current;
}

std::string DecoderBuilder::build_eagle3(const std::string& embeddings) {
    using namespace ov::op;
    OPENVINO_ASSERT(m_cfg.n_layer == 1, "[GGUF] Eagle3 requires one decoder layer");
    const auto feature_width = m_emit.weights().at("fc.weight").get_shape().back();
    auto features = m_emit.add_input("hidden_states", f32, {1, 1, -1, static_cast<int64_t>(feature_width)});
    std::string hidden = "hidden_states";
    if (m_cfg.draft_norm_before_fc)
        hidden = blocks::rms_norm(m_emit, hidden, "enc.output_norm.weight", "eagle3.encoder_norm", m_cfg.rms_eps);
    m_emit.add_weight("fc.weight");
    hidden = m_emit.add_op("GGML_OP_MUL_MAT", "eagle3.fc", {"fc.weight", hidden});
    const std::string p = "blk.0.";
    auto normalized_embedding =
        blocks::rms_norm(m_emit, embeddings, p + "attn_norm.weight", p + "embedding_norm", m_cfg.rms_eps);
    auto normalized_hidden =
        blocks::rms_norm(m_emit, hidden, p + "attn_norm_2.weight", p + "hidden_norm", m_cfg.rms_eps);
    auto joined = m_emit.add_op("GGML_OP_CONCAT",
                                p + "joined",
                                {normalized_embedding, normalized_hidden},
                                0,
                                {{"concat_axis", 0}});
    auto attention = blocks::attention(m_emit, m_cfg, m_kv, 0, joined);
    auto residual = m_cfg.draft_norm_before_residual ? normalized_hidden : hidden;
    auto current = m_emit.add_op("GGML_OP_ADD", p + "attention_residual", {attention, residual});
    auto normalized = blocks::rms_norm(m_emit, current, p + "ffn_norm.weight", p + "ffn_norm", m_cfg.rms_eps);
    auto ffn = blocks::dense_ffn(m_emit, m_cfg, p, normalized);
    current = m_emit.add_op("GGML_OP_ADD", "eagle3.prenorm", {current, ffn});
    m_emit.graph()->model_output_names.push_back(current);
    return current;
}

std::string DecoderBuilder::build_gemma3n(const std::string& embeddings) {
    using namespace ov::op;
    using Value = ov::Output<ov::Node>;
    auto& e = m_emit;
    const auto scalar = [](float value) {
        return v0::Constant::create(f32, {}, {value});
    };
    const auto axis = [](int64_t value) {
        return v0::Constant::create(i64, {1}, {value});
    };
    const auto reshape = [](const Value& x, const std::vector<int64_t>& shape) -> Value {
        return std::make_shared<v1::Reshape>(x, v0::Constant::create(i64, {shape.size()}, shape), true);
    };
    const auto transpose = [](const Value& x, const std::vector<int64_t>& order) -> Value {
        return std::make_shared<v1::Transpose>(x, v0::Constant::create(i64, {order.size()}, order));
    };
    const auto publish = [&](const Value& x, const std::string& name) {
        x.get_node_shared_ptr()->set_friendly_name(name);
        e.add_extra_input_node(name, x.get_node_shared_ptr());
        return name;
    };
    const auto weight = [&](const std::string& name) -> Value {
        e.add_weight(name);
        return e.value(name);
    };
    const auto linear = [&](const Value& x, const std::string& name) -> Value {
        return std::make_shared<v0::MatMul>(x, weight(name + ".weight"), false, true);
    };
    const auto norm = [&](const Value& x, const std::string& name) -> Value {
        return e.value(blocks::rms_norm(e, publish(x, name + ".input"), name + ".weight", name, m_cfg.rms_eps));
    };
    const auto magnitude = [&](const Value& x) -> Value {
        return std::make_shared<v0::Sqrt>(
            std::make_shared<v1::ReduceSum>(std::make_shared<v1::Multiply>(x, x), axis(3), true));
    };
    const auto rescale = [&](const Value& x, const Value& target) -> Value {
        return std::make_shared<v1::Divide>(std::make_shared<v1::Multiply>(x, target), magnitude(x));
    };
    const auto select = [&](const Value& x, int64_t index) -> Value {
        return std::make_shared<v8::Gather>(x, axis(index), v0::Constant::create(i64, {}, {1}));
    };
    const auto router = [&](const Value& x, const std::string& p) -> Value {
        auto normalized = norm(x, p + "altup_router_norm");
        return std::make_shared<v0::Tanh>(
            linear(std::make_shared<v1::Multiply>(normalized, scalar(1.f / m_cfg.n_embd)), p + "altup_router"));
    };
    constexpr int64_t streams = 4;
    auto input = e.value(embeddings);
    auto added = linear(input, "altup_proj");
    added = rescale(added, magnitude(input));
    Value current = std::make_shared<v0::Concat>(ov::OutputVector{input, added}, 1);
    publish(current, "inp_stacked");
    for (int il = 0; il < m_cfg.n_layer; ++il) {
        const auto p = "blk." + std::to_string(il) + ".";
        auto coefficients =
            reshape(linear(router(select(current, 0), p), p + "altup_predict_coef"), {0, -1, streams, streams});
        auto token_major = transpose(current, {0, 2, 1, 3});
        auto prediction = std::make_shared<v0::MatMul>(coefficients, token_major);
        Value predictions = std::make_shared<v1::Add>(current, transpose(prediction, {0, 2, 1, 3}));
        publish(predictions, p + "predictions");
        auto active = select(predictions, 0);
        auto normalized = norm(active, p + "attn_norm");
        auto laurel = norm(linear(linear(normalized, p + "laurel_l"), p + "laurel_r"), p + "laurel_post_norm");
        laurel = std::make_shared<v1::Add>(laurel, normalized);
        auto attention = blocks::attention(e, m_cfg, m_kv, il, publish(normalized, p + "normalized"));
        auto attended = norm(e.value(attention), p + "post_attention_norm");
        attended = std::make_shared<v1::Add>(attended, active);
        Value combined =
            std::make_shared<v1::Multiply>(std::make_shared<v1::Add>(attended, laurel), scalar(1.f / std::sqrt(2.f)));
        publish(combined, p + "attn_laurel");
        auto ffn_input = norm(combined, p + "ffn_norm");
        auto gate = linear(ffn_input, p + "ffn_gate");
        if (il < 10) {
            auto mean = std::make_shared<v1::ReduceMean>(gate, axis(3), true);
            auto delta = std::make_shared<v1::Subtract>(gate, mean);
            auto variance =
                std::make_shared<v1::ReduceSum>(std::make_shared<v1::Multiply>(delta, delta), axis(3), true);
            const auto width = e.weight_rows(p + "ffn_gate.weight");
            auto deviation =
                std::make_shared<v0::Sqrt>(std::make_shared<v1::Multiply>(variance, scalar(1.f / (width - 1))));
            gate = std::make_shared<v0::Relu>(std::make_shared<v1::Subtract>(
                gate,
                std::make_shared<v1::Add>(mean,
                                          std::make_shared<v1::Multiply>(deviation, scalar(1.6448533535003662f)))));
        }
        publish(gate, p + "sparse_gate");
        auto activated_gate = std::make_shared<v7::Gelu>(gate, GeluApproximationMode::TANH);
        auto ffn =
            linear(std::make_shared<v1::Multiply>(activated_gate, linear(ffn_input, p + "ffn_up")), p + "ffn_down");
        auto output = std::make_shared<v1::Add>(norm(ffn, p + "post_ffw_norm"), combined);
        publish(output, p + "attn_ffw_laurel_gated");
        auto innovation = std::make_shared<v1::Subtract>(output, active);
        auto correction = linear(router(output, p), p + "altup_correct_coef");
        correction = std::make_shared<v1::Add>(correction, scalar(1.f));
        correction = transpose(correction, {0, 3, 2, 1});
        Value corrected =
            std::make_shared<v1::Add>(predictions, std::make_shared<v1::Multiply>(innovation, correction));
        publish(corrected, p + "corrected");
        auto corrected_active =
            std::make_shared<v1::Multiply>(select(corrected, 0), weight(p + "altup_correct_scale.weight"));
        auto injection = linear(corrected_active, p + "inp_gate");
        publish(injection, p + "input_projection");
        injection = std::make_shared<v7::Gelu>(injection, GeluApproximationMode::TANH);
        auto per_layer = select(e.value("per_layer_embd"), il);
        injection = norm(linear(std::make_shared<v1::Multiply>(injection, per_layer), p + "proj"), p + "post_norm");
        auto rest = std::make_shared<v8::Gather>(corrected,
                                                 v0::Constant::create(i64, {3}, {1, 2, 3}),
                                                 v0::Constant::create(i64, {}, {1}));
        current = std::make_shared<v0::Concat>(
            ov::OutputVector{select(corrected, 0), std::make_shared<v1::Add>(rest, injection)},
            1);
        publish(current, p + "l_out");
    }
    auto active = select(current, 0);
    auto rest = std::make_shared<v8::Gather>(current,
                                             v0::Constant::create(i64, {3}, {1, 2, 3}),
                                             v0::Constant::create(i64, {}, {1}));
    auto projected = rescale(linear(rest, "altup_unembd_proj"), magnitude(active));
    auto summed = std::make_shared<v1::ReduceSum>(projected, axis(1), true);
    return publish(std::make_shared<v1::Multiply>(std::make_shared<v1::Add>(active, summed), scalar(.25f)),
                   "gemma3n.output");
}

}  // namespace ov::frontend::gguf
