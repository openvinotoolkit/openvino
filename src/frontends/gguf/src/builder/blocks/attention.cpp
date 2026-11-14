// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/blocks/attention.hpp"

#include <map>
#include <vector>

#include "builder/blocks/common.hpp"
#include "builder/blocks/qkv_repack.hpp"

namespace ov::frontend::gguf::blocks {

using ov::element::f32;

namespace {
// Dynamic token length, used for model-input Parameters only.
constexpr int64_t D = -1;
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
    const std::vector<int64_t> merged_shape{0, 1, -1, q_width};
    const std::vector<int64_t> rank3_shape{0, -1, q_width};
    // Gemma4 shared-KV layers have no K/V of their own; newer GGUF files omit those weights.
    const bool has_own_kv = (cfg.shared_kv_layers == 0) || (il < kv.n_own_kv);
    // Qwen35 packs Q and the output gate into one interleaved attn_q_gate projection.
    const bool joint_q_gate = cfg.interleaved_q_gate;
    bool shared_v = false;

    // Q/K/V projection weights. Fused-QKV archs (phi-3, minicpm) carry a single attn_qkv weight;
    // split it into separate q/k/v weights so the rest of the layer is architecture-agnostic.
    if (joint_q_gate) {
        register_interleaved_q_gate(e, cfg, il);
        e.add_weight(p + "attn_k.weight");
        e.add_weight(p + "attn_v.weight");
    } else if (cfg.fused_qkv_override.value_or(e.has_weight(p + "attn_qkv.weight"))) {
        register_fused_qkv(e, cfg, il);
    } else {
        e.add_weight(p + "attn_q.weight");
        if (has_own_kv) {
            e.add_weight(p + "attn_k.weight");
            // MQA tie-V: some global attention layers (e.g. gemma4-12B every 6th layer) have
            // head_count_kv==1 and no separate attn_v.weight; V reuses K's extracted tensors.
            if (e.has_weight(p + "attn_v.weight"))
                e.add_weight(p + "attn_v.weight");
            else {
                e.add_weight_from(p + "attn_v.weight", p + "attn_k");
                shared_v = true;
            }
        }
    }
    std::string q, gate;
    if (joint_q_gate) {
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

    // Q/K/V projection biases (qwen2 / qwen2.5: separate attn_{q,k,v}.bias; fused-QKV archs:
    // attn_qkv.bias, already split by register_fused_qkv above).
    // Qwen35 repacks a joint Q/gate projection; retain its separate bias policy.
    if (cfg.qkv_bias_override.value_or(joint_q_gate ? cfg.has_qkv_bias : e.has_weight(p + "attn_q.bias"))) {
        q = add_bias(e, q, p + "attn_q.bias", p + "Qcur_b");
        if (has_own_kv) {
            k = add_bias(e, k, p + "attn_k.bias", p + "Kcur_b");
            v = add_bias(e, v, p + "attn_v.bias", p + "Vcur_b");
        }
    }

    // Q/K norm: full-width before the head split (OLMoE), or per head before (qwen3, hunyuan,
    // gemma4) or after RoPE.
    const auto qk_norm = [&] {
        q = rms_norm(e, q, p + "attn_q_norm.weight", p + "Qcur_normed", cfg.rms_eps);
        if (has_own_kv)
            k = rms_norm(e, k, p + "attn_k_norm.weight", p + "Kcur_normed", cfg.rms_eps);
    };
    if (cfg.has_qk_norm && cfg.qk_norm_full)
        qk_norm();

    // reshape Q/K/V to [1, n_tokens, n_head(_kv), head_size]
    q = reshape(e, q, {0, -1, n_head_l, head_size_l}, p + "Qcur_r");
    if (has_own_kv) {
        k = reshape(e, k, {0, -1, n_head_kv_l, head_size_l}, p + "Kcur_r");
        v = reshape(e, v, {0, -1, n_head_kv_l, head_size_l}, p + "Vcur_r");
    }

    if (cfg.has_qk_norm && !cfg.qk_norm_full && !cfg.qk_norm_after_rope)
        qk_norm();
    // gemma4: V gets a plain RMSNorm (no multiplicative weight, just normalize).
    if ((cfg.has_v_norm || (cfg.arch == "dflash" && shared_v)) && has_own_kv)
        v = e.add_op("GGML_OP_RMS_NORM", p + "Vcur_normed", {v}, 0, {{"eps", cfg.rms_eps}});

    // RoPE. rope_freqs.weight (per-dim frequency factor) is an optional 3rd input.
    // For gemma4: global layers use rope_freqs (proportional/NTK scaling), SWA layers don't.
    // muse-glimmer ropes ONLY its sliding-window layers; its global layers are NoPE
    // (llama.cpp muse-glimmer.cpp: `const bool use_rope = hparams.is_swa(il)`).
    const bool use_rope =
        (!cfg.rope_on_swa_only || is_swa_layer) && (cfg.rope_skip_period == 0 || (il + 1) % cfg.rope_skip_period != 0);
    if (use_rope) {
        const bool use_rope_freqs = cfg.has_rope_freqs && !is_swa_layer;
        const auto rope = [&](const std::string& x, const std::string& name) {
            std::vector<std::string> inputs{x, "inp_pos"};
            if (use_rope_freqs)
                inputs.emplace_back("rope_freqs.weight");
            return e.add_op("GGML_OP_ROPE", name, inputs, cfg.rope_op_case, {{"rope_config", rope_config_l}});
        };
        q = rope(q, p + "Qcur_rope");
        if (has_own_kv)
            k = rope(k, p + "Kcur_rope");
    }

    if (cfg.has_qk_norm && !cfg.qk_norm_full && cfg.qk_norm_after_rope)
        qk_norm();

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
        temp = reshape(e, temp, {0, -1, 1, 1}, p + "temp_broadcast");
        q = e.add_op("GGML_OP_MUL", p + "Qcur_temp_scaled", {q, temp});
    }

    // ---- KV cache store ----
    // Gemma4: layers with shared_kv_layers have no K/V of their own; they reuse the KV
    // from the last layer of the same SWA type that has its own KV cache. SWA layers
    // reuse the last own-KV SWA layer; global layers reuse the last own-KV global layer.
    int anchor_il = il;
    if (!has_own_kv) {
        anchor_il = is_swa_layer ? kv.anchor_swa : kv.anchor_global;
        if (anchor_il < 0)
            anchor_il = kv.n_own_kv - 1;
    }

    if (!cfg.embedding_model) {
        const std::string kc = "cache_k_l" + std::to_string(anchor_il);
        const std::string vc = "cache_v_l" + std::to_string(anchor_il);
        if (has_own_kv) {
            // Per-layer f16 KV cache Parameters. The K/V read back out of a cache are f16, and so is
            // Q after its convert in the translator, so the FLASH_ATTN inputs agree.
            if (!e.has_model_input(kc)) {
                const ov::PartialShape cache_shape = ps({1, D, n_head_kv_l, head_size_l});
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
            auto& outputs = e.graph()->model_output_names;
            outputs.push_back(kc);
            outputs.push_back(vc);
        } else if (head_size_l < cfg.layer_head_size(anchor_il)) {
            // Shared-KV layer: K/V were already written by the anchor layer's SET_ROWS. A SWA
            // layer sharing a wider global anchor's cache slices it to its own head size, mirroring
            // llama.cpp's ggml_view_4d with n_embd_head_k(il).
            const std::vector<int64_t> hs_slice{3, 0, head_size_l};
            k = e.add_op("GGML_OP_VIEW", p + "k_hslice", {kc}, 3, {{"view_slice", hs_slice}});
            v = e.add_op("GGML_OP_VIEW", p + "v_hslice", {vc}, 3, {{"view_slice", hs_slice}});
        } else {
            k = kc;
            v = vc;
        }
    }

    // q/k/v stay in the ggml-natural [1, n_tokens, n_head(_kv), head_size] layout: the FLASH_ATTN
    // translator permutes AFTER the GQA broadcast of K/V, so Concat -> GQA tile -> Transpose -> SDPA
    // matches the CPU plugin's stateful_sdpa_fusion (ScaledDotProductAttentionWithKVCache).

    // FLASH_ATTN_EXT(q, k, v, mask[, sinks]) -> [1, n_tokens, n_head, head_size].
    // gpt-oss: SWA layers use the sliding-window mask; plus a per-head sink logit.
    std::vector<std::string> attn_in = {q, k, v};
    if (!cfg.embedding_model || cfg.causal_attention)
        attn_in.emplace_back(is_swa_layer ? "self_kq_mask_swa" : "self_kq_mask");
    if (cfg.has_sinks) {
        e.add_weight(p + "attn_sinks.weight");
        attn_in.push_back(p + "attn_sinks.weight");
    }
    std::map<std::string, ov::Any> attn_attrs = {{"scale", kq_scale}};
    if (cfg.embedding_model)
        attn_attrs["f32_attention"] = true;
    if (cfg.attn_soft_cap != 0.0f)
        attn_attrs["kq_soft_cap"] = cfg.attn_soft_cap;
    auto attn = e.add_op("GGML_OP_FLASH_ATTN_EXT",
                         p + "kqv",
                         attn_in,
                         100,  // builder layout: q/k/v are ggml-natural [1, T, n_head, head_size]
                         std::move(attn_attrs));

    // reshape back to [1, 1, n_tokens, n_head*head_size]
    auto attn_2d = reshape(e, attn, merged_shape, p + "kqv_merged");

    if (cfg.has_attn_gate) {
        const auto gate_name = p + (joint_q_gate ? "attn_q.weight" : "attn_gate.weight");
        const auto qtype = e.weight_qtype(gate_name);
        const bool rank3_gate = !gguf_has_u2_weights(qtype);
        auto gate_width = static_cast<size_t>(q_width);
        if (!joint_q_gate) {
            const auto& gate_shape = e.weight_shape(gate_name);
            OPENVINO_ASSERT(gate_shape.size() == 2 && (gate_shape[0] == static_cast<size_t>(n_head_l) ||
                                                       gate_shape[0] == static_cast<size_t>(q_width)),
                            "[GGUF] attention gate must have one value per head or per channel");
            gate_width = gate_shape[0];
        }

        if (joint_q_gate) {
            gate = reshape(e, gate, rank3_gate ? rank3_shape : merged_shape, p + "attn_gate");
        } else {
            e.add_weight(p + "attn_gate.weight");
            gate = e.add_op("GGML_OP_MUL_MAT",
                            p + "attn_gate",
                            {p + "attn_gate.weight", attn_norm},
                            0,
                            {{"activation_rank", rank3_gate ? 3 : 4}});
        }
        gate = e.add_op("GGML_UNARY_OP_SIGMOID", p + "attn_gate_sig", {gate});

        if (gate_width == static_cast<size_t>(n_head_l) && head_size_l > 1) {
            auto heads = reshape(e, attn_2d, {0, -1, n_head_l, head_size_l}, p + "kqv_heads");
            gate = reshape(e, gate, {0, -1, n_head_l, 1}, p + "attn_gate_heads");
            heads = e.add_op("GGML_OP_MUL", p + "kqv_gated_heads", {heads, gate});
            attn_2d = reshape(e, heads, merged_shape, p + "kqv_gated");
        } else {
            if (rank3_gate)
                attn_2d = reshape(e, attn_2d, rank3_shape, p + "kqv_gate_input");
            attn_2d = e.add_op("GGML_OP_MUL", p + "kqv_gated", {attn_2d, gate});
            if (rank3_gate)
                attn_2d = reshape(e, attn_2d, merged_shape, p + "kqv_gated_4d");
        }
    }

    // output projection (+ optional bias)
    e.add_weight(p + "attn_output.weight");
    auto attn_out = e.add_op("GGML_OP_MUL_MAT", p + "attn_out", {p + "attn_output.weight", attn_2d});
    if (cfg.output_bias_override.value_or(joint_q_gate ? cfg.has_attn_out_bias
                                                       : e.has_weight(p + "attn_output.bias"))) {
        attn_out = add_bias(e, attn_out, p + "attn_output.bias", p + "attn_out_b");
    }
    return attn_out;
}

}  // namespace ov::frontend::gguf::blocks
