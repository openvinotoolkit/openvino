// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Architecture detection for the dense/MoE decoder family: turn the parsed GGUF metadata and
// tensor table into the flat DecoderConfig the topology builder reads.
//
// Ground truth for each rule is llama.cpp's per-architecture hparam loading
// (src/models/*.cpp::load_arch_hparams) and llm_graph_context.

#include "decoder_config.hpp"

#include <algorithm>
#include <cmath>
#include <variant>

#include "arch_registry.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace frontend {
namespace gguf {

namespace {

int cfg_int(const std::map<std::string, GGUFMetaData>& config, const std::string& k) {
    auto it = config.find(k);
    OPENVINO_ASSERT(it != config.end(), "[GGUF] internal: missing config key '", k, "'");
    const auto* v = std::get_if<int>(&it->second);
    OPENVINO_ASSERT(v, "[GGUF] internal: config key '", k, "' is not an int");
    return *v;
}

float cfg_float(const std::map<std::string, GGUFMetaData>& config, const std::string& k) {
    auto it = config.find(k);
    OPENVINO_ASSERT(it != config.end(), "[GGUF] internal: missing config key '", k, "'");
    const auto* v = std::get_if<float>(&it->second);
    OPENVINO_ASSERT(v, "[GGUF] internal: config key '", k, "' is not a float");
    return *v;
}

}  // namespace

DecoderConfig::DecoderConfig(const std::map<std::string, GGUFMetaData>& config,
                             const std::unordered_map<std::string, ov::Tensor>& weights) {
    const auto has = [&weights](const std::string& name) {
        return weights.count(name) > 0;
    };
    const auto cfg_i = [&config](const std::string& k) {
        return cfg_int(config, k);
    };
    const auto cfg_f = [&config](const std::string& k) {
        return cfg_float(config, k);
    };

    n_layer = cfg_i("layer_num");
    n_head = cfg_i("head_num");
    n_head_kv = cfg_i("head_num_kv");
    head_size = cfg_i("head_size");
    n_embd = cfg_i("hidden_size");
    rms_eps = cfg_f("rms_norm_eps");
    if (config.count("head_num_kv_per_layer")) {
        if (auto* t = std::get_if<ov::Tensor>(&config.at("head_num_kv_per_layer"))) {
            const auto* data = t->data<uint32_t>();
            n_head_kv_per_layer.assign(data, data + t->get_size());
        }
    }

    arch = std::get<std::string>(config.at("architecture"));

    // Per-architecture structure, auto-detected from the GGUF tensor table (layer 0).
    has_qk_norm = has("blk.0.attn_q_norm.weight");
    // q/k norm width: per-head (head_size, applied after reshape -> qwen3/hunyuan/gemma4)
    // vs full projection width (n_head*head_size, applied before reshape -> OLMoE).
    // For gemma4, norm width equals per-layer head_size (may differ by SWA/global); always
    // per-head. For OLMoE, norm width equals n_head*head_size. Detect by checking against
    // the global head_size; but for gemma4 override to per-head since it has mixed head sizes.
    if (has_qk_norm) {
        const size_t qn = weights.at("blk.0.attn_q_norm.weight").get_shape()[0];
        qk_norm_full = (arch != "gemma4") && (qn != static_cast<size_t>(head_size));
    }
    n_dense_lead = cfg_i("n_layer_dense_lead");
    // Detect MoE from the first non-dense-lead layer (layer 0 may be dense even in MoE models).
    {
        const int probe = std::max(0, n_dense_lead);
        const std::string pp = "blk." + std::to_string(probe) + ".";
        is_moe = has(pp + "ffn_gate_exps.weight");
    }
    // Gemma/Gemma2 use GeGLU (GELU-gated FFN). Detected by arch name since other archs
    // in the supported set (llama, qwen2, qwen3, phi3) all use SwiGLU.
    is_geglu = arch == "gemma" || arch == "gemma2" || arch == "gemma3" || arch == "gemma4";
    // gemma4 only: V projection is also RMSNorm'd (no weight, just normalize). gemma3 has
    // Q/K norm but NO V norm -- llama.cpp src/models/gemma3.cpp norms Qcur/Kcur only and
    // sends Vcur straight to the cache (gemma4.cpp:227 adds the extra ggml_rms_norm on V).
    has_v_norm = (arch == "gemma4");
    n_expert = cfg_i("expert_count");
    n_expert_used = cfg_i("expert_used_count");
    n_expert_shared = cfg_i("expert_shared_count");
    has_moe_gate_bias = has("blk.0.ffn_gate_inp.bias");     // gpt-oss
    has_moe_expert_bias = has("blk.0.ffn_gate_exps.bias");  // gpt-oss
    has_sinks = has("blk.0.attn_sinks.weight");             // gpt-oss
    // Gemma2/Gemma4: per-layer post-attention and post-FFN RMSNorm applied after the sublayer
    // output and before the residual add. Detected from the tensor table at layer 0.
    //
    // Key naming ambiguity across architectures:
    //   Gemma2/Gemma4: attn_norm (pre-attn) + ffn_norm (pre-FFN)
    //                  + post_attention_norm (post-attn) + post_ffw_norm (post-FFN)
    //   gpt-oss:       attn_norm (pre-attn) + post_attention_norm (pre-FFN, no ffn_norm!)
    //   exaone4:       post_attention_norm (pre-attn!) + post_ffw_norm (pre-FFN!) — no attn_norm
    //
    // Rule: post_attention_norm is a true POST-attn norm only when both attn_norm.weight
    // AND ffn_norm.weight also exist.  Same for post_ffw_norm as a true POST-FFN norm.
    {
        const bool has_attn_norm_w = has("blk.0.attn_norm.weight");
        const bool has_ffn_norm_w = has("blk.0.ffn_norm.weight");
        has_attn_post_norm = has("blk.0.post_attention_norm.weight") && has_attn_norm_w && has_ffn_norm_w;
        has_ffn_post_norm = has("blk.0.post_ffw_norm.weight") && has_ffn_norm_w;

        // Compute effective pre-attn and pre-FFN norm key suffixes.
        // Standard: "attn_norm.weight" / "ffn_norm.weight".
        // exaone4:  "post_attention_norm.weight" / "post_ffw_norm.weight".
        // gpt-oss:  "attn_norm.weight" / "post_attention_norm.weight".
        if (!has_attn_norm_w && has("blk.0.post_attention_norm.weight")) {
            attn_norm_key = "post_attention_norm.weight";  // exaone4
        }
        if (!has_ffn_norm_w) {
            if (has("blk.0.post_ffw_norm.weight")) {
                ffn_norm_key = "post_ffw_norm.weight";  // exaone4
            } else if (has("blk.0.post_attention_norm.weight")) {
                ffn_norm_key = "post_attention_norm.weight";  // gpt-oss
            }
        }
    }
    // gpt-oss uses "softmax-after-topk" gating + the OAI gated activation; OLMoE uses
    // softmax-before-topk + plain SwiGLU. Detect by weight-tensor presence so the logic
    // extends to future architectures without touching this file.
    moe_swiglu_oai = has_moe_gate_bias;      // gate_inp bias is OAI-gating-specific
    moe_softmax_weight = has_moe_gate_bias;  // same tensor signals softmax-after-topk
    // SWA is present if sinks (gpt-oss), per-layer flags (gemma4), or has_swa was set
    // from the GGUF metadata (smollm3, exaone-moe, gemma3, and future archs that set
    // attention.sliding_window_pattern / attention.sliding_window).
    has_swa = has_sinks || (arch == "gemma3") || (arch == "gemma4") || cfg_i("has_swa") != 0;
    has_fused_qkv = has("blk.0.attn_qkv.weight");  // phi-3, minicpm
    has_qkv_bias = has("blk.0.attn_q.bias") || has("blk.0.attn_qkv.bias");
    has_attn_out_bias = has("blk.0.attn_output.bias");
    has_rope_freqs = has("rope_freqs.weight");
    // muse-glimmer: sigmoid output gate on the attention sublayer. The gate is projected
    // from the SAME pre-attention normed hidden the Q/K/V projections consume, so it is
    // wired inside the layer right before the output projection.
    has_attn_gate = has("blk.0.attn_gate.weight");
    // muse-glimmer applies a weightless RMSNorm to the token embeddings before layer 0
    // (llama.cpp src/models/muse-glimmer.cpp: build_norm(inpL, nullptr, nullptr, RMS, -1)),
    // ropes only its sliding-window layers (global layers are NoPE), and post-norms with a
    // tighter eps (post_norm_eps = 1e-8) than the pre-norms use.
    const bool is_muse_glimmer = arch == "muse-glimmer";
    scaleless_embd_norm = is_muse_glimmer;
    rope_on_swa_only = is_muse_glimmer;
    post_norm_eps = is_muse_glimmer ? 1e-8f : 0.0f;  // 0 -> reuse rms_eps

    // ---- qwen35 (Qwen3.5/3.6): hybrid Gated-DeltaNet + full attention ----
    // Layers alternate: every full_attention_interval-th layer is full attention, the rest
    // run a linear-attention (GDN) block. llama.cpp src/models/qwen35.cpp.
    is_qwen35 = arch == "qwen35";
    ssm_conv_kernel = cfg_i("ssm_conv_kernel");
    ssm_state_size = cfg_i("ssm_state_size");
    ssm_group_count = cfg_i("ssm_group_count");
    ssm_dt_rank = cfg_i("ssm_time_step_rank");
    ssm_inner_size = cfg_i("ssm_inner_size");
    full_attn_interval = cfg_i("full_attention_interval");
    // NextN/MTP blocks live past the main stack and are not executed in a normal forward
    // pass, so the layer loop must stop before them (llama.cpp runs 0..n_layer(), where
    // n_layer() already excludes them; the GGUF block_count counts them in).
    n_layer_nextn = cfg_i("nextn_predict_layers");
    if (config.count("recurrent_layer_flags")) {
        if (const auto* v = std::get_if<std::vector<int32_t>>(&config.at("recurrent_layer_flags")))
            recurrent_layer_flags = *v;
    }
    if (config.count("rope_sections")) {
        if (const auto* v = std::get_if<std::vector<int32_t>>(&config.at("rope_sections")))
            rope_sections = *v;
    }
    if (is_qwen35) {
        // blk.0 is a RECURRENT layer whose attn_qkv.weight is the GDN q/k/v/z projection, not
        // a fused attention QKV -- the generic probe above would mistake it for phi-3-style
        // fused QKV and try to split it by attention head counts. Likewise blk.0's
        // attn_gate.weight is the GDN output gate (z), not an attention gate.
        has_fused_qkv = false;
        // qwen35's full-attention layers DO have a sigmoid attention output gate, and it is
        // the same construction muse-glimmer uses -- projected from the pre-attention normed
        // hidden, sigmoid'd, multiplied into the merged attention output before wo. The only
        // difference is where the weight lives: qwen35 interleaves it per head inside attn_q
        // rather than storing a separate tensor, so register_qwen35_q_gate() de-interleaves it
        // into blk.N.attn_q.weight + blk.N.attn_gate.weight and the shared path handles it.
        has_attn_gate = true;
        // Per-head QK-norm on the full-attention layers. Auto-detection probes blk.0, which is
        // recurrent and carries no attn_q_norm, so set it explicitly.
        has_qk_norm = true;
        qk_norm_full = false;
        // qwen35's pre-FFN norm key ("post_attention_norm.weight", HF's
        // post_attention_layernorm) already falls out of the generic rule above: attn_norm
        // exists, ffn_norm does not, which is the gpt-oss pattern.
        // MTP/NextN blocks are stored past the main stack and are not part of a normal
        // forward pass, so the layer loop must not walk into them.
        n_layer -= n_layer_nextn;
        OPENVINO_ASSERT(n_layer > 0, "[GGUF] qwen35: no trunk layers left after excluding NextN blocks");
    }

    // Per-architecture scalars from metadata (1.0 / 0.0 when absent -> no-op).
    embedding_scale = cfg_f("embedding_scale");
    residual_scale = cfg_f("residual_scale");
    logit_scale = cfg_f("logit_scale");
    attention_scale = cfg_f("attention_scale");            // 0 -> 1/sqrt(head_size)
    expert_weights_scale = cfg_f("expert_weights_scale");  // 0 -> 1.0 no-op
    rope_freq_base_swa = cfg_f("rope_freq_base_swa");
    swa_layer_pattern = cfg_i("swa_layer_pattern");
    // Gemma4: per-layer SWA boolean flags (non-empty when swa_layer_pattern==0).
    if (config.count("swa_layer_flags"))
        swa_layer_flags = std::get<std::vector<int32_t>>(config.at("swa_layer_flags"));
    attn_soft_cap = cfg_f("attn_logit_softcapping");          // 0 -> no soft-cap
    final_logit_soft_cap = cfg_f("final_logit_softcapping");  // 0 -> no soft-cap
    // Gemma4: per-layer input embeddings and shared KV layers.
    n_embd_per_layer = cfg_i("n_embd_per_layer");
    shared_kv_layers = cfg_i("shared_kv_layers");
    // Gemma4: SWA layers use a smaller head size than global attention layers.
    head_size_swa = cfg_i("head_size_swa");
    rope_dim_swa = cfg_i("rope_dimension_count_swa");

    // Interleaved M-RoPE (qwen35 / qwen3vl) is its own mode, so it must be decided here rather
    // than in the per-arch block above -- this assignment runs later and would overwrite it.
    rope_op_case = is_qwen35                   ? ROPE_OP_CASE_IMROPE
                   : arch_uses_neox_rope(arch) ? ROPE_OP_CASE_NEOX
                                               : ROPE_OP_CASE_NORMAL;

    rope_config.n_dims = cfg_i("rope_dimension_count");
    rope_config.n_ctx_orig = cfg_i("rope_n_ctx_orig");
    rope_config.freq_base = cfg_f("rope_freq_base");
    rope_config.freq_scale = cfg_f("rope_freq_scale");
    rope_config.ext_factor = cfg_f("rope_ext_factor");
    rope_config.attn_factor = 1.0f;
    rope_config.beta_fast = 32.0f;
    rope_config.beta_slow = 1.0f;
    // Gemma4: separate rope config for SWA layers (different freq_base and n_dims).
    rope_config_swa = rope_config;
    rope_config_swa.freq_base = cfg_f("rope_freq_base_swa");
    rope_config_swa.n_dims = cfg_i("rope_dimension_count_swa");
    // Per-op sin/cos is required whenever SWA and global layers differ in any rope
    // parameter that feeds the shared sin/cos table: n_dims (gemma4) OR freq_base (gemma3,
    // whose SWA layers rope at 10000 vs the global 1000000). Without it every layer would
    // share the global table and SWA layers would be roped wrong.
    const bool swa_dims_differ = rope_dim_swa > 0 && rope_dim_swa != rope_config.n_dims;
    const bool swa_freq_differs = rope_config_swa.freq_base != rope_config.freq_base;
    if (has_swa && (swa_dims_differ || swa_freq_differs)) {
        use_per_op_rope = true;
    }
    // M-RoPE: inp_pos carries 4 sections per token and only the first n_dims of each head
    // are rotated (qwen35: head_size 256, rope.dimension_count 64). Build sin/cos per ROPE op
    // rather than from the shared table, whose layout assumes the single-section full-head
    // case (see RopeConfig::is_imrope / use_per_op_rope).
    if (rope_op_case == ROPE_OP_CASE_IMROPE) {
        rope_config.is_imrope = true;
        use_per_op_rope = true;
    }
}

bool DecoderConfig::layer_is_swa(int il) const {
    if (!swa_layer_flags.empty()) {
        return il < static_cast<int>(swa_layer_flags.size()) && swa_layer_flags[il] != 0;
    }
    return has_swa && swa_layer_pattern > 0 && (il % swa_layer_pattern < (swa_layer_pattern - 1));
}

int DecoderConfig::layer_head_size(int il) const {
    return (layer_is_swa(il) && head_size_swa > 0) ? head_size_swa : head_size;
}

int DecoderConfig::layer_n_head_kv(int il) const {
    return (!n_head_kv_per_layer.empty() && il < static_cast<int>(n_head_kv_per_layer.size()))
               ? static_cast<int>(n_head_kv_per_layer[il])
               : n_head_kv;
}

float DecoderConfig::layer_kq_scale(int il) const {
    return attention_scale != 0.0f ? attention_scale : 1.0f / std::sqrt(static_cast<float>(layer_head_size(il)));
}

RopeConfig DecoderConfig::layer_rope_config(int il) const {
    return layer_is_swa(il) ? rope_config_swa : rope_config;
}

bool DecoderConfig::is_recurrent_layer(int il) const {
    if (!is_qwen35) {
        return false;
    }
    if (il < static_cast<int>(recurrent_layer_flags.size())) {
        return recurrent_layer_flags[il] != 0;
    }
    return full_attn_interval > 0 && ((il + 1) % full_attn_interval != 0);
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
