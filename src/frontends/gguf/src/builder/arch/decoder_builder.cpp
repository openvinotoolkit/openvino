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

#include "builder/api/metadata_store.hpp"
#include "builder/blocks/common.hpp"
#include "builder/blocks/ffn.hpp"
#include "builder/blocks/gated_delta_net.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace ov::frontend::gguf {

using ov::element::f32;
using ov::element::i32;
using ov::element::i64;

namespace {
// Dynamic token length, used for model-input Parameters only.
constexpr int64_t D = -1;

// Families drafted like llama.cpp's NextN MTP graphs: enorm/hnorm -> eh_proj -> one regular block
// -> shared_head_norm/output_norm -> shared_head_head/output. Add a family once its NextN block is
// verified to be a regular layer of that family against the ggml MTP graph.
bool has_mtp_draft(const std::string& architecture) {
    return architecture == "qwen35";
}

// The draft uses the first NextN block (llama.cpp: n_layer() + nextn_layer_offset 0).
int nextn_layer(const DecoderParameters& parameters) {
    return parameters.dimensions.layers - parameters.hybrid.nextn_layers;
}

std::string nextn_tensor(int layer, const std::string& name) {
    return "blk." + std::to_string(layer) + ".nextn." + name + ".weight";
}

template <typename MetadataMap>
bool hadamard_tied_output(const MetadataMap& metadata) {
    const auto it = metadata.find("prism.hadamard.tied_output");
    if (it == metadata.end())
        return false;
    const auto* tensor = std::get_if<ov::Tensor>(&it->second);
    OPENVINO_ASSERT(tensor && tensor->get_element_type() == ov::element::boolean && tensor->get_size() == 1,
                    "[GGUF] invalid prism.hadamard.tied_output");
    return *tensor->template data<const char>() != 0;
}

DecoderParameters component_parameters(const BuildContext& context, DecoderParameters parameters, bool draft) {
    OPENVINO_ASSERT(context.component.empty() || context.component == "mtp",
                    "[GGUF] unknown decoder component: ",
                    context.component);
    if (context.component.empty())
        return parameters;
    OPENVINO_ASSERT(has_mtp_draft(parameters.architecture),
                    "[GGUF] MTP is not supported for architecture: ",
                    parameters.architecture);
    OPENVINO_ASSERT(!parameters.embedding_model, "[GGUF] MTP requires a generative decoder");
    if (!draft)
        return parameters;
    OPENVINO_ASSERT(parameters.hybrid.nextn_layers == 1 && parameters.dimensions.layers > 1,
                    "[GGUF] MTP requires exactly one NextN block and at least one target layer");
    const auto layer = nextn_layer(parameters);
    OPENVINO_ASSERT(!context.weights->weights.count(nextn_tensor(layer, "embed_tokens")),
                    "[GGUF] MTP requires shared token embeddings; a separate NextN embedding table is not supported");
    const auto target_layers = static_cast<size_t>(layer);
    const auto stored_layers = static_cast<size_t>(parameters.dimensions.layers);
    OPENVINO_ASSERT(parameters.layers.empty() || parameters.layers.size() == target_layers ||
                        parameters.layers.size() == stored_layers,
                    "[GGUF] per-layer settings must describe every decoder layer");
    parameters.layers.resize(stored_layers);

    auto& kv_heads = parameters.dimensions.kv_heads_per_layer;
    const bool infer_nextn_kv_heads = kv_heads.size() == target_layers;
    if (infer_nextn_kv_heads)
        kv_heads.push_back(1);
    if (parameters.attention.sliding_layers.size() == target_layers)
        parameters.attention.sliding_layers.push_back(0);
    if (parameters.hybrid.recurrent_layers.size() == target_layers)
        parameters.hybrid.recurrent_layers.push_back(0);

    parameters.hybrid.nextn_layers = 0;
    // NextN blocks are attention blocks, also in hybrid recurrent families.
    parameters.layers[layer].recurrent = false;
    if (infer_nextn_kv_heads) {
        const auto key = "blk." + std::to_string(layer) + ".attn_k.weight";
        const auto weight = context.weights->weights.find(key);
        OPENVINO_ASSERT(weight != context.weights->weights.end() && weight->second.shape.size() == 2,
                        "[GGUF] cannot infer NextN KV heads from ",
                        key);
        // The temporary nonzero KV count lets DecoderConfig resolve attention-only SWA.
        const auto head_size = DecoderConfig(parameters, context.weights->weights).layer_head_size(layer);
        const auto rows = weight->second.shape[0];
        OPENVINO_ASSERT(
            rows > 0 && rows % static_cast<size_t>(head_size) == 0 &&
                rows / static_cast<size_t>(head_size) <= static_cast<size_t>(std::numeric_limits<int>::max()),
            "[GGUF] invalid NextN KV head dimensions for ",
            key);
        kv_heads.back() = static_cast<uint32_t>(rows / static_cast<size_t>(head_size));
    }
    parameters.model_contract.sequence_inputs = {"hidden_states"};
    return parameters;
}
}  // namespace

DecoderBuilder::DecoderBuilder(const BuildContext& context, const DecoderParameters& parameters)
    : DecoderBuilder(context, parameters, nullptr) {}

DecoderBuilder::DecoderBuilder(const BuildContext& context,
                               const DecoderParameters& parameters,
                               const GraphEmitter* target)
    : m_context(context),
      m_parameters(parameters),
      m_cfg(component_parameters(context, parameters, target != nullptr), context.weights->weights),
      m_emit(context.weights->weights, context.arch, context.weights->translators),
      m_kv(blocks::KvCachePlan::build(m_cfg)),
      m_draft(target != nullptr),
      m_nextn_layer(m_draft ? nextn_layer(parameters) : -1),
      m_export_hidden(!context.component.empty()) {
    if (m_draft)
        m_emit.share_weights_from(*target);
    if (m_draft || parameters.hybrid.nextn_layers > 0) {
        // NextN weights belong to the draft: it keeps their Hadamard names plus its shared
        // embeddings and head, and the target keeps every other name.
        auto metadata = detail::MetadataAccess::get(context.metadata).map;
        const auto first_nextn = nextn_layer(parameters);
        const auto in_nextn = [&](const std::string& name) {
            for (int layer = first_nextn; layer < parameters.dimensions.layers; ++layer)
                if (name.rfind("blk." + std::to_string(layer) + ".", 0) == 0)
                    return true;
            return false;
        };
        const auto head = lm_head_weight();
        bool rotated = false;
        for (const auto& key : {"prism.hadamard.weight_names", "prism.hadamard.inverse_weight_names"}) {
            if (const auto it = metadata.find(key); it != metadata.end()) {
                auto& names = std::get<std::vector<std::string>>(it->second);
                names.erase(std::remove_if(names.begin(),
                                           names.end(),
                                           [&](const auto& name) {
                                               return m_draft ? name != "token_embd.weight" && name != head &&
                                                                    !in_nextn(name)
                                                              : in_nextn(name);
                                           }),
                            names.end());
                rotated = rotated || !names.empty();
            }
        }
        // An untied model that uses no rotated weight has no Hadamard transform.
        if (!rotated && !hadamard_tied_output(metadata)) {
            for (auto it = metadata.begin(); it != metadata.end();)
                it = it->first.rfind("prism.hadamard.", 0) == 0 ? metadata.erase(it) : std::next(it);
        }
        const detail::MetadataStore store{metadata};
        m_cfg.configure_graph(m_emit, GgufMetadata(store), m_draft);
    } else {
        m_cfg.configure_graph(m_emit, context.metadata);
    }
    if (m_export_hidden)
        m_emit.graph()->model_contract.sequence_outputs.push_back("last_hidden_state");
    m_emit.graph()->component = context.component;
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
    // GET_ROWS(token_embd.weight, inp_tokens) -> "embd"
    m_emit.add_weight("token_embd.weight");
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

    if (m_cfg.is_recurrent_layer(il)) {
        // Hybrid stack: 3 of every 4 layers replace attention with a Gated DeltaNet block.
        // It produces the same thing the attention path does -- the sublayer output before
        // the residual -- so the shared FFN tail below is reused verbatim.
        std::string gdn_out = blocks::gated_delta_net(m_emit, m_cfg, il, attn_norm);
        if (m_cfg.residual_scale != 1.0f) {
            gdn_out = blocks::scale(m_emit, gdn_out, m_cfg.residual_scale, p + "attn_out_scaled");
        }
        std::string sa = inpSA;
        std::string ao = gdn_out;
        if (selects_output_rows(il)) {
            ao = m_emit.add_op("GGML_OP_GET_ROWS", p + "attn_out_g", {gdn_out, "inp_out_ids"});
            sa = m_emit.add_op("GGML_OP_GET_ROWS", p + "inpSA_g", {inpSA, "inp_out_ids"});
        }
        auto ffn_inp_r = m_emit.add_op("GGML_OP_ADD", p + "ffn_inp", {ao, sa});
        auto ffn_norm_r = norm(m_emit, ffn_inp_r, p + m_cfg.ffn_norm_key, p + "ffn_norm", m_cfg.rms_eps);
        // Mirror the non-recurrent FFN dispatch below: Qwen3-Next is MoE for every trunk layer
        // (llama.cpp routes both the GDN and full-attention layers through build_moe_ffn), so a
        // real model has no per-layer dense FFN tensors here and dense_ffn's weight lookup
        // would assert.
        const bool is_moe_layer_r = m_cfg.layer_is_moe(il);
        auto down_r = is_moe_layer_r   ? blocks::moe_ffn(m_emit, m_cfg, p, ffn_norm_r)
                      : m_cfg.is_geglu ? blocks::geglu_ffn(m_emit, m_cfg, p, ffn_norm_r)
                                       : blocks::dense_ffn(m_emit, m_cfg, p, ffn_norm_r);
        if (m_cfg.residual_scale != 1.0f) {
            down_r = blocks::scale(m_emit, down_r, m_cfg.residual_scale, p + "ffn_out_scaled");
        }
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
    auto ffn_norm =
        m_cfg.post_norm_only ? ffn_inp : norm(m_emit, ffn_inp, p + m_cfg.ffn_norm_key, p + "ffn_norm", m_cfg.rms_eps);

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
        m_emit.add_weight(p + "layer_output_scale.weight");
        cur = m_emit.add_op("GGML_OP_MUL", p + "scaled_out", {cur, p + "layer_output_scale.weight"});
    }
    return cur;
}

std::string DecoderBuilder::nextn_weight(const std::string& name) const {
    return nextn_tensor(m_nextn_layer, name);
}

std::string DecoderBuilder::lm_head_weight() const {
    if (m_draft && m_emit.has_weight(nextn_weight("shared_head_head")))
        return nextn_weight("shared_head_head");
    return m_emit.has_weight("output.weight") ? "output.weight" : "token_embd.weight";
}

std::string DecoderBuilder::build_head(const std::string& in) {
    // final norm + lm_head
    const auto norm = m_cfg.layer_norm ? blocks::layer_norm : blocks::rms_norm;
    const auto norm_weight = m_draft && m_emit.has_weight(nextn_weight("shared_head_norm"))
                                 ? nextn_weight("shared_head_norm")
                                 : std::string("output_norm.weight");
    std::string cur = norm(m_emit, in, norm_weight, "result_norm", m_cfg.rms_eps);
    if (m_export_hidden) {
        (*m_emit.graph()->values)["last_hidden_state"] = m_emit.value(cur);
        m_emit.graph()->model_output_names.push_back("last_hidden_state");
        cur = m_emit.add_op("GGML_OP_GET_ROWS", "result_norm_selected", {cur, "inp_out_ids"});
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

    const auto lm_head_w = lm_head_weight();
    m_emit.add_weight(lm_head_w);
    auto logits = m_emit.add_op("GGML_OP_MUL_MAT", "result_output", {lm_head_w, cur});
    // MiniCPM scales the logits (1/(n_embd/dim_model_base)).
    if (m_cfg.logit_scale != 1.0f) {
        logits = blocks::scale(m_emit, logits, m_cfg.logit_scale, "result_output_scaled");
    }
    // Gemma2/Gemma3 final logit soft-cap: tanh(logits / cap) * cap.
    if (m_cfg.final_logit_soft_cap != 0.0f) {
        logits = blocks::scale(m_emit, logits, 1.0f / m_cfg.final_logit_soft_cap, "logits_softcap_scaled");
        logits = m_emit.add_op("GGML_UNARY_OP_TANH", "logits_softcap_tanh", {logits});
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

    if (m_draft) {
        auto hidden = m_emit.add_input("hidden_states", f32, ps({D, D, m_cfg.n_embd}));
        auto lifted = std::make_shared<ov::op::v0::Unsqueeze>(hidden, ov::op::v0::Constant::create(i64, {1}, {1}));
        (*m_emit.graph()->values)["mtp_hidden"] = lifted;
        auto h = blocks::rms_norm(m_emit, "mtp_hidden", nextn_weight("hnorm"), "mtp_hnorm", m_cfg.rms_eps);
        auto e = blocks::rms_norm(m_emit, cur, nextn_weight("enorm"), "mtp_enorm", m_cfg.rms_eps);
        cur = m_emit.add_op("GGML_OP_CONCAT", "mtp_concat", {e, h}, 0, {{"concat_axis", 0}});
        m_emit.add_weight(nextn_weight("eh_proj"));
        cur = m_emit.add_op("GGML_OP_MUL_MAT", "mtp_eh_proj", {nextn_weight("eh_proj"), cur});
    }
    const int first_layer = m_draft ? m_nextn_layer : 0;
    const int end_layer = m_draft ? m_nextn_layer + 1 : m_cfg.n_layer;
    for (int il = first_layer; il < end_layer; ++il) {
        cur = build_layer(il, cur);
    }

    const std::string logits = build_head(cur);
    m_emit.validate_hadamard();

    // logits is the primary output; the cache_k/v_l* outputs (appended per layer above)
    // become Assign sinks via MakeStateful.
    auto graph = m_emit.graph();
    graph->model_output_names.insert(graph->model_output_names.begin(), logits);
    if (!m_draft && m_export_hidden && m_parameters.hybrid.nextn_layers > 0)
        graph->mtp_draft = DecoderBuilder(m_context, m_parameters, &m_emit).build();
    return graph;
}

}  // namespace ov::frontend::gguf
