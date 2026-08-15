// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/blocks/attention.hpp"

#include <map>
#include <vector>

#include "builder/blocks/common.hpp"
#include "builder/blocks/qkv_repack.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace blocks {

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
                      const std::string& attn_norm,
                      int64_t T) {
    const std::string p = "blk." + std::to_string(il) + ".";
    auto& graph = *e.graph();

    // Per-layer hyperparameters (SWA vs global head size / rope / scale; per-layer KV heads).
    // See the DecoderConfig accessors for the arch-specific derivation rules.
    const bool is_swa_layer = cfg.layer_is_swa(il);
    const int head_size_l = cfg.layer_head_size(il);
    const int n_head_kv_l = cfg.layer_n_head_kv(il);
    const float kq_scale = cfg.layer_kq_scale(il);
    const RopeConfig rope_config_l = cfg.layer_rope_config(il);

    // Q/K/V projections: MUL_MAT(w, attn_norm), then conceptual reshape to heads.
    // Fused-QKV archs (phi-3, minicpm) carry a single attn_qkv weight; split it into
    // separate q/k/v weights so the rest of the layer is architecture-agnostic.
    if (cfg.is_qwen35) {
        register_qwen35_q_gate(e, cfg, il);
        e.add_weight(p + "attn_k.weight");
        e.add_weight(p + "attn_v.weight");
    } else if (cfg.has_fused_qkv) {
        register_fused_qkv(e, cfg, il);
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
    auto q = e.add_op("GGML_OP_MUL_MAT",
                      p + "Qcur",
                      {p + "attn_q.weight", attn_norm},
                      ps({1, 1, T, head_size_l * cfg.n_head}),
                      f32);
    auto k = e.add_op("GGML_OP_MUL_MAT",
                      p + "Kcur",
                      {p + "attn_k.weight", attn_norm},
                      ps({1, 1, T, head_size_l * n_head_kv_l}),
                      f32);
    auto v = e.add_op("GGML_OP_MUL_MAT",
                      p + "Vcur",
                      {p + "attn_v.weight", attn_norm},
                      ps({1, 1, T, head_size_l * n_head_kv_l}),
                      f32);

    // Q/K/V projection biases (qwen2 / qwen2.5: separate attn_{q,k,v}.bias).
    if (cfg.has_qkv_bias && !cfg.has_fused_qkv) {
        q = add_bias(e, q, p + "attn_q.bias", p + "Qcur_b");
        k = add_bias(e, k, p + "attn_k.bias", p + "Kcur_b");
        v = add_bias(e, v, p + "attn_v.bias", p + "Vcur_b");
    }

    // Full-width q/k norm (OLMoE): normalize the whole projection before splitting heads.
    if (cfg.has_qk_norm && cfg.qk_norm_full) {
        q = rms_norm(e, q, p + "attn_q_norm.weight", p + "Qcur_normed", cfg.rms_eps);
        k = rms_norm(e, k, p + "attn_k_norm.weight", p + "Kcur_normed", cfg.rms_eps);
    }

    // reshape Q/K/V to [1, n_tokens, n_head(_kv), head_size]
    q = e.add_op("GGML_OP_RESHAPE", p + "Qcur_r", {q}, ps({1, T, cfg.n_head, head_size_l}), f32, 1);
    k = e.add_op("GGML_OP_RESHAPE", p + "Kcur_r", {k}, ps({1, T, n_head_kv_l, head_size_l}), f32, 1);
    v = e.add_op("GGML_OP_RESHAPE", p + "Vcur_r", {v}, ps({1, T, n_head_kv_l, head_size_l}), f32, 1);

    // per-head q_norm / k_norm (qwen3, hunyuan, gemma4)
    if (cfg.has_qk_norm && !cfg.qk_norm_full) {
        q = rms_norm(e, q, p + "attn_q_norm.weight", p + "Qcur_normed", cfg.rms_eps);
        k = rms_norm(e, k, p + "attn_k_norm.weight", p + "Kcur_normed", cfg.rms_eps);
    }
    // gemma4: V gets a plain RMSNorm (no multiplicative weight, just normalize).
    if (cfg.has_v_norm) {
        v = e.add_op("GGML_OP_RMS_NORM",
                     p + "Vcur_normed",
                     {v},
                     ps({1, T, n_head_kv_l, head_size_l}),
                     f32,
                     0,
                     {{"eps", cfg.rms_eps}});
    }

    // RoPE (NEOX). rope_freqs.weight (per-dim frequency factor) is an optional 3rd input.
    // For gemma4: global layers use rope_freqs (proportional/NTK scaling), SWA layers don't.
    // muse-glimmer ropes ONLY its sliding-window layers; its global layers are NoPE
    // (llama.cpp muse-glimmer.cpp: `const bool use_rope = hparams.is_swa(il)`).
    const bool use_rope = !cfg.rope_on_swa_only || is_swa_layer;
    const bool use_rope_freqs = cfg.has_rope_freqs && !is_swa_layer;
    const std::vector<std::string> q_rope_in = use_rope_freqs
                                                   ? std::vector<std::string>{q, "inp_pos", "rope_freqs.weight"}
                                                   : std::vector<std::string>{q, "inp_pos"};
    const std::vector<std::string> k_rope_in = use_rope_freqs
                                                   ? std::vector<std::string>{k, "inp_pos", "rope_freqs.weight"}
                                                   : std::vector<std::string>{k, "inp_pos"};
    if (use_rope) {
        q = e.add_op("GGML_OP_ROPE",
                     p + "Qcur_rope",
                     q_rope_in,
                     ps({1, T, cfg.n_head, head_size_l}),
                     f32,
                     cfg.rope_op_case,
                     {{"rope_config", rope_config_l}});
        k = e.add_op("GGML_OP_ROPE",
                     p + "Kcur_rope",
                     k_rope_in,
                     ps({1, T, n_head_kv_l, head_size_l}),
                     f32,
                     cfg.rope_op_case,
                     {{"rope_config", rope_config_l}});
    }

    // ---- KV cache store ----
    // Gemma4: layers with shared_kv_layers have no K/V of their own; they reuse the KV
    // from the last layer of the same SWA type that has its own KV cache. SWA layers
    // reuse the last own-KV SWA layer; global layers reuse the last own-KV global layer.
    const bool has_own_kv = (cfg.shared_kv_layers == 0) || (il < kv.n_own_kv);
    int anchor_il = il;
    if (!has_own_kv) {
        anchor_il = is_swa_layer ? kv.anchor_swa : kv.anchor_global;
        if (anchor_il < 0)
            anchor_il = kv.n_own_kv - 1;  // fallback
    }
    const std::string kc = "cache_k_l" + std::to_string(anchor_il);
    const std::string vc = "cache_v_l" + std::to_string(anchor_il);

    if (has_own_kv) {
        // Per-layer f16 KV cache Parameters. The K/V read back out of a cache are f16, and so is
        // Q after its convert in the translator, so the FLASH_ATTN inputs agree.
        const ov::PartialShape cache_shape = ps({1, D, n_head_kv_l, head_size_l});
        if (!e.has_model_input(kc)) {
            e.add_input(kc, ov::element::f16, cache_shape);
            e.add_input(vc, ov::element::f16, cache_shape);
        }
        e.set_tensor_meta(kc, ps({1, T, n_head_kv_l, head_size_l}), ov::element::f16);
        e.set_tensor_meta(vc, ps({1, T, n_head_kv_l, head_size_l}), ov::element::f16);

        // SET_ROWS(cur, idx, cache) -> the cache with this step's rows written in. Lowered by
        // the frontend to a stateless ScatterUpdate, or by the caller-registered MakeStateful
        // extension to a ReadValue/Concat/Assign OpenVINO state.
        k = e.add_op("GGML_OP_SET_ROWS",
                     kc,
                     {k, "inp_kv_idx", kc},
                     ps({1, T, n_head_kv_l, head_size_l}),
                     ov::element::f16);
        v = e.add_op("GGML_OP_SET_ROWS",
                     vc,
                     {v, "inp_kv_idx", vc},
                     ps({1, T, n_head_kv_l, head_size_l}),
                     ov::element::f16);
        // The written-through caches are model outputs, so the stateless graph returns each
        // updated cache as a Result (which MakeStateful, when registered, turns into an Assign
        // sink paired with the cache's ReadValue).
        graph.model_output_names.push_back(kc);
        graph.model_output_names.push_back(vc);
    } else {
        // Shared-KV layer: K/V have already been set in the anchor layer's SET_ROWS.
        // Use the anchor's combined cache. If the current layer has a smaller head size
        // (SWA shared layer vs a global anchor), slice K/V to the layer's head_size along
        // the last dim, mirroring llama.cpp's ggml_view_4d with n_embd_head_k(il).
        const ov::PartialShape& anchor_kc_shape = e.shape_of_tensor(kc);  // [1, T, n_kv, anchor_head]
        const int64_t anchor_hs = anchor_kc_shape[3].is_static() ? anchor_kc_shape[3].get_length() : cfg.head_size;
        if (head_size_l < static_cast<int>(anchor_hs)) {
            // Shrink the last (head-size) axis to head_size_l. This is a plain single-axis shrink
            // at offset 0, which is exactly what the shared VIEW op_case 3 does -- and what the
            // cgraph decoder assigns to llama.cpp's corresponding ggml_view_4d with
            // n_embd_head_k(il) -- so describe it the way that case expects rather than with a
            // builder-only case: "view_slice" = {ov_axis, start, len} plus the input's own shape.
            // No "view_reshape": the sliced shape already is this node's output shape.
            const ov::PartialShape slice_shape = ps({1, T, n_head_kv_l, head_size_l});
            const std::vector<int64_t> hs_slice{3, 0, int64_t(head_size_l)};
            k = e.add_op("GGML_OP_VIEW",
                         p + "k_hslice",
                         {kc},
                         slice_shape,
                         ov::element::f16,
                         3,
                         {{"view_slice", hs_slice}, {"input_ggml_shape", e.static_shape_of(kc)}});
            v = e.add_op("GGML_OP_VIEW",
                         p + "v_hslice",
                         {vc},
                         slice_shape,
                         ov::element::f16,
                         3,
                         {{"view_slice", hs_slice}, {"input_ggml_shape", e.static_shape_of(vc)}});
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
    std::vector<std::string> attn_in = {q, k, v, mask_name};
    if (cfg.has_sinks) {
        e.add_named_weight(p + "attn_sinks.weight");
        attn_in.push_back(p + "attn_sinks.weight");
    }
    std::map<std::string, ov::Any> attn_attrs = {{"scale", kq_scale}};
    if (cfg.attn_soft_cap != 0.0f) {
        attn_attrs["kq_soft_cap"] = cfg.attn_soft_cap;
    }
    auto attn = e.add_op("GGML_OP_FLASH_ATTN_EXT",
                         p + "kqv",
                         attn_in,
                         ps({1, T, cfg.n_head, head_size_l}),
                         f32,
                         100,  // builder layout: q/k/v are ggml-natural [1, T, n_head, head_size]
                         std::move(attn_attrs));

    // reshape back to [1, 1, n_tokens, n_head*head_size]
    auto attn_2d =
        e.add_op("GGML_OP_RESHAPE", p + "kqv_merged", {attn}, ps({1, 1, T, cfg.n_head * head_size_l}), f32, 2);

    // muse-glimmer: sigmoid output gate. The gate is a projection of the PRE-attention
    // normed hidden (the same `attn_norm` tensor Q/K/V come from), squashed by sigmoid and
    // multiplied elementwise into the merged attention output before the wo projection.
    if (cfg.has_attn_gate) {
        e.add_weight(p + "attn_gate.weight");
        auto gate = e.add_op("GGML_OP_MUL_MAT",
                             p + "attn_gate",
                             {p + "attn_gate.weight", attn_norm},
                             ps({1, 1, T, cfg.n_head * head_size_l}),
                             f32);
        gate = e.add_op("GGML_UNARY_OP_SIGMOID", p + "attn_gate_sig", {gate}, e.shape_of_tensor(gate), f32);
        attn_2d = e.add_op("GGML_OP_MUL", p + "kqv_gated", {attn_2d, gate}, e.shape_of_tensor(attn_2d), f32);
    }

    // output projection (+ optional bias)
    e.add_weight(p + "attn_output.weight");
    auto attn_out = e.add_op("GGML_OP_MUL_MAT",
                             p + "attn_out",
                             {p + "attn_output.weight", attn_2d},
                             ps({1, 1, T, cfg.n_embd}),
                             f32);
    if (cfg.has_attn_out_bias) {
        attn_out = add_bias(e, attn_out, p + "attn_output.bias", p + "attn_out_b");
    }
    return attn_out;
}

}  // namespace blocks
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
