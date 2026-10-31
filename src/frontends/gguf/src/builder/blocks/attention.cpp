// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/blocks/attention.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <vector>

#include "builder/blocks/common.hpp"
#include "builder/blocks/qkv_repack.hpp"

namespace ov::frontend::gguf::blocks {

using ov::element::f32;

namespace {
// Dynamic token length, used for model-input Parameters only.
constexpr int64_t D = -1;

// Block-sparse attention over head-major K/V states [1, kv_heads, keys, head_size] that grow by
// concatenation. QSAPrepare keeps the pooled keys of the complete blocks and the incomplete block
// as states, so no index state grows with every token. q/k/v are [1, T, heads, head_size].
std::string sparse_attention(GraphEmitter& e,
                             const DecoderConfig& cfg,
                             int il,
                             const std::string& attn_norm,
                             const std::string& q,
                             const std::string& k,
                             const std::string& v,
                             float kq_scale) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::string l = "_l" + std::to_string(il);
    auto& graph = *e.graph();
    const auto& sparse = cfg.sparse;
    const int64_t block = sparse.key_block_size;
    const int64_t index_dim = sparse.index_head_size;
    auto rope = cfg.layer_rope_config(il);
    rope.is_imrope = rope.is_imrope || (cfg.rope_op_case >> 16) == 2;
    const int64_t rotary = rope.n_dims;
    OPENVINO_ASSERT(rotary > 0 && rotary % 2 == 0 && rotary <= index_dim,
                    "[GGUF] sparse attention index heads must contain the rotary dimensions");

    // Shared by every sparse layer: valid key counts and the RoPE tables of the current tokens.
    if (!graph.values->count("qsa_key_length"))
        e.add_op("GGUF_OP_ATTENTION_KEY_LENGTH", "qsa_key_length", {"self_kq_mask"});
    if (!graph.values->count("qsa_rope_cos"))
        e.add_op("GGUF_OP_ROPE_CACHE", "qsa_rope_cos", {"inp_pos"}, 0, {{"rope_config", rope}}, {"qsa_rope_sin"});

    for (const auto* name :
         {"indexer.q_proj.weight", "indexer.k_proj.weight", "indexer.q_norm.weight", "indexer.k_norm.weight"})
        e.add_weight(p + name);
    const auto index_query = e.add_op("GGML_OP_MUL_MAT", p + "index_q", {p + "indexer.q_proj.weight", attn_norm});
    const auto index_key = e.add_op("GGML_OP_MUL_MAT", p + "index_k", {p + "indexer.k_proj.weight", attn_norm});
    OPENVINO_ASSERT(e.value(index_key).get_partial_shape()[3] == index_dim &&
                        e.value(index_query).get_partial_shape()[3] == index_dim * sparse.index_heads,
                    "[GGUF] sparse attention indexer projections do not match the indexer heads in layer ",
                    il);

    const std::string block_keys = "qsa_block_keys" + l, pending_keys = "qsa_pending_keys" + l,
                      pending_rope = "qsa_pending_rope" + l;
    e.add_input(block_keys, f32, ps({1, D, index_dim}));
    e.add_input(pending_keys, f32, ps({1, block - 1, index_dim}));
    e.add_input(pending_rope, f32, ps({1, 2 * rotary}));
    const auto prepared = e.add_op("GGUF_OP_QSA_PREPARE",
                                   block_keys + "_out",
                                   {index_key,
                                    block_keys,
                                    "qsa_key_length",
                                    "qsa_rope_cos",
                                    "qsa_rope_sin",
                                    p + "indexer.k_norm.weight",
                                    pending_keys,
                                    pending_rope},
                                   0,
                                   {{"key_block_size", block}, {"eps", cfg.rms_eps}},
                                   {pending_keys + "_out", pending_rope + "_out"});
    for (const auto& state : {block_keys, pending_keys, pending_rope}) {
        graph.model_output_names.push_back(state + "_out");
        graph.recurrent_states.emplace_back(state, state + "_out");
    }
    const auto selected =
        e.add_op("GGUF_OP_SPARSE_ATTENTION_INDEXER",
                 p + "qsa_blocks",
                 {index_query, p + "indexer.q_norm.weight", "qsa_rope_cos", "qsa_rope_sin", prepared, "qsa_key_length"},
                 0,
                 {{"heads", int64_t{sparse.index_heads}},
                  {"key_block_size", block},
                  {"token_budget", int64_t{sparse.token_budget}},
                  {"eps", cfg.rms_eps}},
                 {p + "qsa_block_count"});

    const int64_t kv_heads = cfg.layer_n_head_kv(il), head_size = cfg.layer_head_size(il);
    std::array<std::string, 2> caches;
    const std::array<std::string, 2> current{k, v};
    for (size_t i = 0; i < 2; ++i) {
        const std::string cache = (i == 0 ? "qsa_cache_k" : "qsa_cache_v") + l;
        e.add_input(cache, f32, ps({1, kv_heads, D, head_size}));
        const auto head_major =
            e.add_op("GGML_OP_PERMUTE", cache + "_cur", {current[i]}, 1, {{"perm", std::vector<int64_t>{0, 2, 1, 3}}});
        caches[i] = e.add_op("GGML_OP_CONCAT", cache + "_out", {cache, head_major}, 0, {{"concat_axis", int{1}}});
        graph.model_output_names.push_back(caches[i]);
        graph.recurrent_states.emplace_back(cache, caches[i]);
    }
    return e.add_op("GGUF_OP_SPARSE_ATTENTION",
                    p + "kqv",
                    {q, caches[0], caches[1], selected, p + "qsa_block_count", "qsa_key_length"},
                    0,
                    {{"key_block_size", block}, {"scale", kq_scale}});
}
}  // namespace

KvCachePlan KvCachePlan::build(const DecoderConfig& cfg) {
    KvCachePlan plan;
    plan.n_own_kv = cfg.n_layer - cfg.shared_kv_layers;
    if (cfg.shared_kv_layers > 0) {
        for (int i = plan.n_own_kv - 1; i >= 0; --i) {
            const bool i_is_swa = cfg.layer_is_swa(i);
            if (plan.anchor_swa < 0 && i_is_swa)
                plan.anchor_swa = i;
            if (plan.anchor_global < 0 && !i_is_swa)
                plan.anchor_global = i;
            if (plan.anchor_swa >= 0 && plan.anchor_global >= 0)
                break;
        }
    }
    return plan;
}

std::string attention(GraphEmitter& e,
                      const DecoderConfig& cfg,
                      const KvCachePlan& kv,
                      int il,
                      const std::string& attn_norm) {
    const std::string p = "blk." + std::to_string(il) + ".";
    auto& graph = *e.graph();

    // Per-layer hyperparameters (SWA vs global head size / rope / scale; per-layer KV heads).
    // See the DecoderConfig accessors for the arch-specific derivation rules.
    const bool is_swa_layer = cfg.layer_is_swa(il);
    const int head_size_l = cfg.layer_head_size(il);
    const int n_head_kv_l = cfg.layer_n_head_kv(il);
    const int n_head_l = cfg.layer_n_head(il);
    const float kq_scale = cfg.layer_kq_scale(il);
    const RopeConfig rope_config_l = cfg.layer_rope_config(il);
    // Widen before multiplying the reshape dimensions.
    const int64_t q_width = static_cast<int64_t>(head_size_l) * n_head_l;
    // Gemma4 shared-KV layers have no K/V of their own; newer GGUF files omit those weights.
    const bool has_own_kv = (cfg.shared_kv_layers == 0) || (il < kv.n_own_kv);

    // Q/K/V projections: MUL_MAT(w, attn_norm), then conceptual reshape to heads.
    // Fused-QKV archs (phi-3, minicpm) carry a single attn_qkv weight; split it into
    // separate q/k/v weights so the rest of the layer is architecture-agnostic.
    if (cfg.interleaved_q_gate) {
        register_interleaved_q_gate(e, cfg, il);
        e.add_weight(p + "attn_k.weight");
        e.add_weight(p + "attn_v.weight");
    } else if (cfg.fused_qkv_override.value_or(e.has_weight(p + "attn_qkv.weight"))) {
        register_fused_qkv(e, cfg, il);
    } else if (!has_own_kv) {
        e.add_weight(p + "attn_q.weight");
    } else {
        e.add_weight(p + "attn_q.weight");
        e.add_weight(p + "attn_k.weight");
        // MQA tie-V: some global attention layers (e.g. gemma4-12B every 6th layer) have
        // head_count_kv==1 and no separate attn_v.weight; V shares K's weight tensor.
        if (e.has_weight(p + "attn_v.weight")) {
            e.add_weight(p + "attn_v.weight");
        } else {
            // MQA tie-V: emit a separate attn_v.weight GGML_OP_NONE leaf that reuses K's
            // extracted tensors, so attn_v.weight resolves in the tensor map like any weight.
            e.add_weight_from(p + "attn_v.weight", p + "attn_k");
        }
    }
    std::string q, gate;
    if (e.weight_emitted(p + "attn_q_gate.weight")) {
        const auto joint = e.add_op("GGML_OP_MUL_MAT", p + "q_gate", {p + "attn_q_gate.weight", attn_norm});
        const auto projections =
            split_interleaved_projection(e, joint, {p + "Qcur", p + "attn_gate_joint"}, n_head_l, head_size_l);
        q = projections[0];
        gate = projections[1];
    } else {
        q = e.add_op("GGML_OP_MUL_MAT", p + "Qcur", {p + "attn_q.weight", attn_norm});
    }
    std::string k, v;
    if (has_own_kv) {
        k = e.add_op("GGML_OP_MUL_MAT", p + "Kcur", {p + "attn_k.weight", attn_norm});
        v = e.add_op("GGML_OP_MUL_MAT", p + "Vcur", {p + "attn_v.weight", attn_norm});
    }

    // Q/K/V projection biases (qwen2 / qwen2.5: separate attn_{q,k,v}.bias; phi-3-style
    // fused-QKV archs: attn_qkv.bias, already split into attn_{q,k,v}.bias by
    // register_fused_qkv above).
    // Qwen35 repacks a joint Q/gate projection; retain its separate bias policy.
    if (cfg.qkv_bias_override.value_or(cfg.interleaved_q_gate ? cfg.has_qkv_bias : e.has_weight(p + "attn_q.bias"))) {
        q = add_bias(e, q, p + "attn_q.bias", p + "Qcur_b");
        if (has_own_kv) {
            k = add_bias(e, k, p + "attn_k.bias", p + "Kcur_b");
            v = add_bias(e, v, p + "attn_v.bias", p + "Vcur_b");
        }
    }

    // Full-width q/k norm (OLMoE): normalize the whole projection before splitting heads.
    if (cfg.has_qk_norm && cfg.qk_norm_full) {
        q = rms_norm(e, q, p + "attn_q_norm.weight", p + "Qcur_normed", cfg.rms_eps);
        if (has_own_kv)
            k = rms_norm(e, k, p + "attn_k_norm.weight", p + "Kcur_normed", cfg.rms_eps);
    }

    // reshape Q/K/V to [1, n_tokens, n_head(_kv), head_size]
    q = e.add_op("GGML_OP_RESHAPE",
                 p + "Qcur_r",
                 {q},
                 6,
                 {{"reshape_target", std::vector<int64_t>{0, -1, n_head_l, head_size_l}}, {"special_zero", true}});
    if (has_own_kv) {
        k = e.add_op(
            "GGML_OP_RESHAPE",
            p + "Kcur_r",
            {k},
            6,
            {{"reshape_target", std::vector<int64_t>{0, -1, n_head_kv_l, head_size_l}}, {"special_zero", true}});
        v = e.add_op(
            "GGML_OP_RESHAPE",
            p + "Vcur_r",
            {v},
            6,
            {{"reshape_target", std::vector<int64_t>{0, -1, n_head_kv_l, head_size_l}}, {"special_zero", true}});
    }

    // per-head q_norm / k_norm (qwen3, hunyuan, gemma4)
    if (cfg.has_qk_norm && !cfg.qk_norm_full && !cfg.qk_norm_after_rope) {
        q = rms_norm(e, q, p + "attn_q_norm.weight", p + "Qcur_normed", cfg.rms_eps);
        if (has_own_kv)
            k = rms_norm(e, k, p + "attn_k_norm.weight", p + "Kcur_normed", cfg.rms_eps);
    }
    // gemma4: V gets a plain RMSNorm (no multiplicative weight, just normalize).
    if (cfg.has_v_norm && has_own_kv) {
        v = e.add_op("GGML_OP_RMS_NORM", p + "Vcur_normed", {v}, 0, {{"eps", cfg.rms_eps}});
    }

    // RoPE (NEOX). rope_freqs.weight (per-dim frequency factor) is an optional 3rd input.
    // For gemma4: global layers use rope_freqs (proportional/NTK scaling), SWA layers don't.
    // muse-glimmer ropes ONLY its sliding-window layers; its global layers are NoPE
    // (llama.cpp muse-glimmer.cpp: `const bool use_rope = hparams.is_swa(il)`).
    const bool use_rope =
        (!cfg.rope_on_swa_only || is_swa_layer) && (cfg.rope_skip_period == 0 || (il + 1) % cfg.rope_skip_period != 0);
    const bool use_rope_freqs = cfg.has_rope_freqs && !is_swa_layer;
    const std::vector<std::string> q_rope_in = use_rope_freqs
                                                   ? std::vector<std::string>{q, "inp_pos", "rope_freqs.weight"}
                                                   : std::vector<std::string>{q, "inp_pos"};
    const std::vector<std::string> k_rope_in = use_rope_freqs
                                                   ? std::vector<std::string>{k, "inp_pos", "rope_freqs.weight"}
                                                   : std::vector<std::string>{k, "inp_pos"};
    if (use_rope) {
        q = e.add_op("GGML_OP_ROPE", p + "Qcur_rope", q_rope_in, cfg.rope_op_case, {{"rope_config", rope_config_l}});
        if (has_own_kv)
            k = e.add_op("GGML_OP_ROPE",
                         p + "Kcur_rope",
                         k_rope_in,
                         cfg.rope_op_case,
                         {{"rope_config", rope_config_l}});
    }

    if (cfg.has_qk_norm && !cfg.qk_norm_full && cfg.qk_norm_after_rope) {
        q = rms_norm(e, q, p + "attn_q_norm.weight", p + "Qcur_normed", cfg.rms_eps);
        if (has_own_kv)
            k = rms_norm(e, k, p + "attn_k_norm.weight", p + "Kcur_normed", cfg.rms_eps);
    }

    if (cfg.attention_temperature_scale != 0.0f) {
        auto temp = e.add_op("GGML_OP_CPY", p + "temp_pos", {"inp_pos"}, 0, {{"dst_type", f32}});
        temp = scale(e, temp, 1.0f / cfg.rope_config.n_ctx_orig, p + "temp_window");
        // Positions are nonnegative: integer conversion implements floor(position / context).
        temp = e.add_op("GGML_OP_CPY", p + "temp_floor", {temp}, 0, {{"dst_type", ov::element::i32}});
        temp = e.add_op("GGML_OP_CPY", p + "temp_float", {temp}, 0, {{"dst_type", f32}});
        temp = e.add_op("GGML_OP_SCALE", p + "temp_offset", {temp}, 0, {{"bias", 1.0f}});
        temp = e.add_op("GGML_OP_LOG", p + "temp_log", {temp});
        temp = e.add_op("GGML_OP_SCALE",
                        p + "temp_scale",
                        {temp},
                        0,
                        {{"scale", cfg.attention_temperature_scale}, {"bias", 1.0f}});
        temp = e.add_op("GGML_OP_RESHAPE",
                        p + "temp_broadcast",
                        {temp},
                        6,
                        {{"reshape_target", std::vector<int64_t>{0, -1, 1, 1}}, {"special_zero", true}});
        q = e.add_op("GGML_OP_MUL", p + "Qcur_temp_scaled", {q, temp});
    }

    std::string attn;
    if (cfg.layer_is_sparse(il)) {
        attn = sparse_attention(e, cfg, il, attn_norm, q, k, v, kq_scale);
    } else {
        // ---- KV cache store ----
        // Gemma4: layers with shared_kv_layers have no K/V of their own; they reuse the KV
        // from the last layer of the same SWA type that has its own KV cache. SWA layers
        // reuse the last own-KV SWA layer; global layers reuse the last own-KV global layer.
        int anchor_il = il;
        if (!has_own_kv) {
            anchor_il = is_swa_layer ? kv.anchor_swa : kv.anchor_global;
            if (anchor_il < 0)
                anchor_il = kv.n_own_kv - 1;  // fallback
        }
        const std::string kc = "cache_k_l" + std::to_string(anchor_il);
        const std::string vc = "cache_v_l" + std::to_string(anchor_il);

        if (!cfg.embedding_model && has_own_kv) {
            // Per-layer f16 KV cache Parameters. The K/V read back out of a cache are f16, and so is
            // Q after its convert in the translator, so the FLASH_ATTN inputs agree.
            const ov::PartialShape cache_shape = ps({1, D, n_head_kv_l, head_size_l});
            if (!e.has_model_input(kc)) {
                e.add_input(kc, ov::element::f16, cache_shape);
                e.add_input(vc, ov::element::f16, cache_shape);
            }

            // SET_ROWS(cur, idx, cache) -> the cache with this step's rows written in. Lowered by
            // the frontend to a stateless ScatterUpdate, or by the caller-registered MakeStateful
            // extension to a ReadValue/Concat/Assign OpenVINO state.
            k = e.add_op("GGML_OP_SET_ROWS", kc, {k, "inp_kv_idx", kc});
            v = e.add_op("GGML_OP_SET_ROWS", vc, {v, "inp_kv_idx", vc});
            // The written-through caches are model outputs, so the stateless graph returns each
            // updated cache as a Result (which MakeStateful, when registered, turns into an Assign
            // sink paired with the cache's ReadValue).
            graph.model_output_names.push_back(kc);
            graph.model_output_names.push_back(vc);
        } else if (!cfg.embedding_model) {
            // Shared-KV layer: K/V have already been set in the anchor layer's SET_ROWS.
            // Use the anchor's combined cache. If the current layer has a smaller head size
            // (SWA shared layer vs a global anchor), slice K/V to the layer's head_size along
            // the last dim, mirroring llama.cpp's ggml_view_4d with n_embd_head_k(il).
            const ov::PartialShape& anchor_kc_shape = e.value(kc).get_partial_shape();  // [1, T, n_kv, anchor_head]
            const int64_t anchor_hs = anchor_kc_shape[3].is_static() ? anchor_kc_shape[3].get_length() : cfg.head_size;
            if (head_size_l < static_cast<int>(anchor_hs)) {
                // Slice the shared cache to this layer's head width.
                const std::vector<int64_t> hs_slice{3, 0, int64_t(head_size_l)};
                k = e.add_op("GGML_OP_VIEW", p + "k_hslice", {kc}, 3, {{"view_slice", hs_slice}});
                v = e.add_op("GGML_OP_VIEW", p + "v_hslice", {vc}, 3, {{"view_slice", hs_slice}});
            } else {
                k = kc;
                v = vc;
            }
        }

        // NOTE: q/k/v stay in the ggml-natural [1, n_tokens, n_head(_kv), head_size] layout
        // here. The PERMUTE to [1, n_head, n_tokens, head_size] is done INSIDE the FLASH_ATTN
        // translator, AFTER the GQA broadcast of K/V. That ordering -- Concat -> GQA tile ->
        // single Transpose -> SDPA -- is the exact shape the CPU plugin's stateful_sdpa_fusion
        // matches (its multi-query-bcst sits on the concat output, before one transpose), so
        // the attention fuses into ScaledDotProductAttentionWithKVCache. Permuting before the
        // tile (the old order) put the transpose between concat and tile and blocked the fuse.

        // FLASH_ATTN_EXT(q, k, v, mask[, sinks]) -> [1, n_tokens, n_head, head_size].
        // gpt-oss: SWA layers use the sliding-window mask; plus a per-head sink logit.
        const std::string mask_name = is_swa_layer ? "self_kq_mask_swa" : "self_kq_mask";
        std::vector<std::string> attn_in = {q, k, v};
        if (!cfg.embedding_model || cfg.causal_attention)
            attn_in.push_back(mask_name);
        if (cfg.has_sinks) {
            e.add_named_weight(p + "attn_sinks.weight");
            attn_in.push_back(p + "attn_sinks.weight");
        }
        std::map<std::string, ov::Any> attn_attrs = {{"scale", kq_scale}};
        if (cfg.embedding_model)
            attn_attrs["f32_attention"] = true;
        if (cfg.attn_soft_cap != 0.0f) {
            attn_attrs["kq_soft_cap"] = cfg.attn_soft_cap;
        }
        attn = e.add_op("GGML_OP_FLASH_ATTN_EXT",
                        p + "kqv",
                        attn_in,
                        100,  // builder layout: q/k/v are ggml-natural [1, T, n_head, head_size]
                        std::move(attn_attrs));
    }

    // reshape back to [1, 1, n_tokens, n_head*head_size]
    auto attn_2d = e.add_op("GGML_OP_RESHAPE",
                            p + "kqv_merged",
                            {attn},
                            6,
                            {{"reshape_target", std::vector<int64_t>{0, 1, -1, q_width}}, {"special_zero", true}});

    if (cfg.has_attn_gate) {
        bool rank3_gate;
        if (!gate.empty()) {
            rank3_gate = e.weight_tensor(p + "attn_q.weight").get_element_type() != ov::element::u2;
            gate =
                e.add_op("GGML_OP_RESHAPE",
                         p + "attn_gate",
                         {gate},
                         6,
                         {{"reshape_target",
                           rank3_gate ? std::vector<int64_t>{0, -1, q_width} : std::vector<int64_t>{0, 1, -1, q_width}},
                          {"special_zero", true}});
        } else {
            e.add_weight(p + "attn_gate.weight");
            const auto gate_weight = std::find_if(graph.nodes.begin(), graph.nodes.end(), [&](const auto& node) {
                return node.name == p + "attn_gate.weight";
            });
            rank3_gate = gate_weight->attributes.at("gguf.blob.weight").template as<ov::Tensor>().get_element_type() !=
                         ov::element::u2;
            gate = e.add_op("GGML_OP_MUL_MAT",
                            p + "attn_gate",
                            {p + "attn_gate.weight", attn_norm},
                            0,
                            {{"activation_rank", rank3_gate ? 3 : 4}});
        }
        gate = e.add_op("GGML_UNARY_OP_SIGMOID", p + "attn_gate_sig", {gate});
        auto gate_width = static_cast<size_t>(q_width);
        if (!cfg.interleaved_q_gate) {
            const auto& gate_shape = e.weight_tensor(p + "attn_gate.weight").get_shape();
            OPENVINO_ASSERT(gate_shape.size() == 2 && (gate_shape[0] == static_cast<size_t>(n_head_l) ||
                                                       gate_shape[0] == static_cast<size_t>(q_width)),
                            "[GGUF] attention gate must have one value per head or per channel");
            gate_width = gate_shape[0];
        }
        if (gate_width == static_cast<size_t>(n_head_l) && head_size_l > 1) {
            auto heads = e.add_op(
                "GGML_OP_RESHAPE",
                p + "kqv_heads",
                {attn_2d},
                6,
                {{"reshape_target", std::vector<int64_t>{0, -1, n_head_l, head_size_l}}, {"special_zero", true}});
            gate = e.add_op("GGML_OP_RESHAPE",
                            p + "attn_gate_heads",
                            {gate},
                            6,
                            {{"reshape_target", std::vector<int64_t>{0, -1, n_head_l, 1}}, {"special_zero", true}});
            heads = e.add_op("GGML_OP_MUL", p + "kqv_gated_heads", {heads, gate});
            attn_2d = e.add_op("GGML_OP_RESHAPE",
                               p + "kqv_gated",
                               {heads},
                               6,
                               {{"reshape_target", std::vector<int64_t>{0, 1, -1, q_width}}, {"special_zero", true}});
        } else {
            const auto gate_attention =
                rank3_gate
                    ? e.add_op("GGML_OP_RESHAPE",
                               p + "kqv_gate_input",
                               {attn_2d},
                               6,
                               {{"reshape_target", std::vector<int64_t>{0, -1, q_width}}, {"special_zero", true}})
                    : attn_2d;
            const auto gated = e.add_op("GGML_OP_MUL", p + "kqv_gated", {gate_attention, gate});
            attn_2d =
                rank3_gate
                    ? e.add_op("GGML_OP_RESHAPE",
                               p + "kqv_gated_4d",
                               {gated},
                               6,
                               {{"reshape_target", std::vector<int64_t>{0, 1, -1, q_width}}, {"special_zero", true}})
                    : gated;
        }
    }

    // output projection (+ optional bias)
    e.add_weight(p + "attn_output.weight");
    auto attn_out = e.add_op("GGML_OP_MUL_MAT", p + "attn_out", {p + "attn_output.weight", attn_2d});
    if (cfg.output_bias_override.value_or(cfg.interleaved_q_gate ? cfg.has_attn_out_bias
                                                                 : e.has_weight(p + "attn_output.bias"))) {
        attn_out = add_bias(e, attn_out, p + "attn_output.bias", p + "attn_out_b");
    }
    return attn_out;
}

}  // namespace ov::frontend::gguf::blocks
