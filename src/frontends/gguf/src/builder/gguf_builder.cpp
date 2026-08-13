// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Native GGUF -> GgufGraph builder. Emits nodes in the GGML op vocabulary that reproduce
// llama.cpp's cgraph topology, so the resulting GgufGraph drives the same op translators as
// the llama.cpp cgraph path. No llama.cpp / gguf dependency.
//
// One generic dense-transformer builder covers the whole "llama-family" of architectures
// (llama-3, qwen2/2.5, qwen3, phi-3, minicpm, hunyuan-dense, ...). Per-architecture
// differences are derived almost entirely from the GGUF itself:
//   - presence of per-head q/k norm weights  (blk.N.attn_{q,k}_norm.weight)  -> qwen3, hunyuan
//   - presence of q/k/v projection biases     (blk.N.attn_{q,k,v}.bias)       -> qwen2/2.5
//   - presence of an output-projection bias   (blk.N.attn_output.bias)
//   - presence of rope frequency factors      (rope_freqs.weight)            -> llama-3, phi-3
//   - scalar scales from metadata             (embedding/residual/logit/attention scale)
//                                                                            -> minicpm
// so adding a new architecture of this family is just adding its name to kSupportedArchs.
// Structurally novel families (MoE routing, SSM, fused-QKV-only, etc.) still need new code.
//
// Ground truth for the topology: llama.cpp src/models/{llama,qwen2,qwen3,phi3,minicpm,
// hunyuan-*}.cpp and the build_norm / build_qkv / build_attn / build_attn_mha / build_ffn
// expansions in src/llama-graph.cpp, plus the KV-cache cpy_k/get_k (SET_ROWS + VIEW) in
// src/llama-kv-cache.cpp. Op-case values follow ggml-decoder.cpp::compute_op_case.
//
// Builds the STATELESS graph: KV caches are plain model Parameters written by GGML_OP_SET_ROWS and
// read back as Results, and every activation keeps ggml's rank-4 shape. A caller that wants an
// OpenVINO KV cache registers ov::frontend::gguf::pass::MakeStateful as a transformation extension.

#include "gguf_builder.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <set>

#include "gguf_graph.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/util/log.hpp"
#include "quant/gguf.hpp"
#include "quant/weights.hpp"

namespace ov {
namespace frontend {
namespace gguf {

namespace {

// gguf ROPE op-case (see ggml-decoder.cpp::compute_op_case). The high 16 bits encode the
// mode: NORMAL=0 (llama/minicpm: rotate consecutive pairs), NEOX=1 (qwen/phi/hunyuan:
// rotate halves). The input is not a VIEW here so the low bits stay 0.
constexpr int ROPE_OP_CASE_NORMAL = 0x00000000;
constexpr int ROPE_OP_CASE_NEOX = 0x00010000;
// IMROPE=2: interleaved multimodal rope (qwen35 / qwen3vl). inp_pos carries 4 mrope sections.
constexpr int ROPE_OP_CASE_IMROPE = 0x00020000;

// Architectures whose rope is NEOX (rotate-halves); everything else in the supported set
// uses NORMAL (rotate consecutive pairs). Mirrors llama_model_rope_type.
bool arch_uses_neox_rope(const std::string& arch) {
    return arch == "qwen2" || arch == "qwen3" || arch == "phi3" || arch == "hunyuan-dense" || arch == "gpt-oss" ||
           arch == "gemma" || arch == "gemma2" || arch == "gemma3" || arch == "gemma4" || arch == "olmoe" ||
           // H2 2025 dense additions
           arch == "exaone4" || arch == "plamo3" || arch == "mellum" ||
           // H2 2025 MoE additions
           arch == "hunyuan-moe" || arch == "glm4moe" || arch == "bailingmoe2" || arch == "exaone-moe" ||
           arch == "minimax-m2" ||
           // H2 2025 VL backbone / other
           arch == "jais2" || arch == "deepseek2-ocr";
}

// Shapes are kept in the OpenVINO/GGML logical order [ne3, ne2, ne1, ne0] (reverse of GGUF
// on-disk order), matching the decoder's get_shape(). The translators consume them as-is.
// Token length is dynamic (-1) in dim ne1.

class TransformerBuilder {
public:
    TransformerBuilder(const std::map<std::string, GGUFMetaData>& config,
                       std::unordered_map<std::string, ov::Tensor>& weights,
                       std::unordered_map<std::string, gguf_tensor_type>& qtypes)
        : m_config(config),
          m_weights(weights),
          m_qtypes(qtypes) {
        m_graph = std::make_shared<GgufGraph>();

        m_n_layer = cfg_int("layer_num");
        m_n_head = cfg_int("head_num");
        m_n_head_kv = cfg_int("head_num_kv");
        m_head_size = cfg_int("head_size");
        m_n_embd = cfg_int("hidden_size");
        m_rms_eps = cfg_float("rms_norm_eps");
        if (config.count("head_num_kv_per_layer")) {
            if (auto* t = std::get_if<ov::Tensor>(&config.at("head_num_kv_per_layer"))) {
                const auto* data = t->data<uint32_t>();
                m_n_head_kv_per_layer.assign(data, data + t->get_size());
            }
        }

        const std::string arch_str = std::get<std::string>(config.at("architecture"));

        // Per-architecture structure, auto-detected from the GGUF tensor table (layer 0).
        m_has_qk_norm = m_weights.count("blk.0.attn_q_norm.weight") > 0;
        // q/k norm width: per-head (head_size, applied after reshape -> qwen3/hunyuan/gemma4)
        // vs full projection width (n_head*head_size, applied before reshape -> OLMoE).
        // For gemma4, norm width equals per-layer head_size (may differ by SWA/global); always
        // per-head. For OLMoE, norm width equals n_head*head_size. Detect by checking against
        // the global head_size; but for gemma4 override to per-head since it has mixed head sizes.
        if (m_has_qk_norm) {
            const size_t qn = weight_tensor("blk.0.attn_q_norm.weight").get_shape()[0];
            m_qk_norm_full = (arch_str != "gemma4") && (qn != static_cast<size_t>(m_head_size));
        }
        m_n_dense_lead = cfg_int("n_layer_dense_lead");
        // Detect MoE from the first non-dense-lead layer (layer 0 may be dense even in MoE models).
        {
            const int probe = std::max(0, m_n_dense_lead);
            const std::string pp = "blk." + std::to_string(probe) + ".";
            m_is_moe = m_weights.count(pp + "ffn_gate_exps.weight") > 0;
        }
        // Gemma/Gemma2 use GeGLU (GELU-gated FFN). Detected by arch name since other archs
        // in the supported set (llama, qwen2, qwen3, phi3) all use SwiGLU.
        m_is_geglu = arch_str == "gemma" || arch_str == "gemma2" || arch_str == "gemma3" || arch_str == "gemma4";
        // gemma4 only: V projection is also RMSNorm'd (no weight, just normalize). gemma3 has
        // Q/K norm but NO V norm -- llama.cpp src/models/gemma3.cpp norms Qcur/Kcur only and
        // sends Vcur straight to the cache (gemma4.cpp:227 adds the extra ggml_rms_norm on V).
        m_has_v_norm = (arch_str == "gemma4");
        m_n_expert = cfg_int("expert_count");
        m_n_expert_used = cfg_int("expert_used_count");
        m_n_expert_shared = cfg_int("expert_shared_count");
        m_has_moe_gate_bias = m_weights.count("blk.0.ffn_gate_inp.bias") > 0;     // gpt-oss
        m_has_moe_expert_bias = m_weights.count("blk.0.ffn_gate_exps.bias") > 0;  // gpt-oss
        m_has_sinks = m_weights.count("blk.0.attn_sinks.weight") > 0;             // gpt-oss
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
            const bool has_attn_norm_w = m_weights.count("blk.0.attn_norm.weight") > 0;
            const bool has_ffn_norm_w = m_weights.count("blk.0.ffn_norm.weight") > 0;
            m_has_attn_post_norm =
                m_weights.count("blk.0.post_attention_norm.weight") > 0 && has_attn_norm_w && has_ffn_norm_w;
            m_has_ffn_post_norm = m_weights.count("blk.0.post_ffw_norm.weight") > 0 && has_ffn_norm_w;

            // Compute effective pre-attn and pre-FFN norm key suffixes.
            // Standard: "attn_norm.weight" / "ffn_norm.weight".
            // exaone4:  "post_attention_norm.weight" / "post_ffw_norm.weight".
            // gpt-oss:  "attn_norm.weight" / "post_attention_norm.weight".
            if (!has_attn_norm_w && m_weights.count("blk.0.post_attention_norm.weight") > 0) {
                m_attn_norm_key = "post_attention_norm.weight";  // exaone4
            }
            if (!has_ffn_norm_w) {
                if (m_weights.count("blk.0.post_ffw_norm.weight") > 0) {
                    m_ffn_norm_key = "post_ffw_norm.weight";  // exaone4
                } else if (m_weights.count("blk.0.post_attention_norm.weight") > 0) {
                    m_ffn_norm_key = "post_attention_norm.weight";  // gpt-oss
                }
            }
        }
        // gpt-oss uses "softmax-after-topk" gating + the OAI gated activation; OLMoE uses
        // softmax-before-topk + plain SwiGLU. Detect by weight-tensor presence so the logic
        // extends to future architectures without touching this file.
        m_moe_swiglu_oai = m_has_moe_gate_bias;      // gate_inp bias is OAI-gating-specific
        m_moe_softmax_weight = m_has_moe_gate_bias;  // same tensor signals softmax-after-topk
        // SWA is present if sinks (gpt-oss), per-layer flags (gemma4), or has_swa was set
        // from the GGUF metadata (smollm3, exaone-moe, gemma3, and future archs that set
        // attention.sliding_window_pattern / attention.sliding_window).
        m_has_swa = m_has_sinks || (arch_str == "gemma3") || (arch_str == "gemma4") || cfg_int("has_swa") != 0;
        m_has_fused_qkv = m_weights.count("blk.0.attn_qkv.weight") > 0;  // phi-3, minicpm
        m_has_qkv_bias = m_weights.count("blk.0.attn_q.bias") > 0 || m_weights.count("blk.0.attn_qkv.bias") > 0;
        m_has_attn_out_bias = m_weights.count("blk.0.attn_output.bias") > 0;
        m_has_rope_freqs = m_weights.count("rope_freqs.weight") > 0;
        // muse-glimmer: sigmoid output gate on the attention sublayer. The gate is projected
        // from the SAME pre-attention normed hidden the Q/K/V projections consume, so it is
        // wired inside the layer right before the output projection (see build()).
        m_has_attn_gate = m_weights.count("blk.0.attn_gate.weight") > 0;
        // muse-glimmer applies a weightless RMSNorm to the token embeddings before layer 0
        // (llama.cpp src/models/muse-glimmer.cpp: build_norm(inpL, nullptr, nullptr, RMS, -1)),
        // ropes only its sliding-window layers (global layers are NoPE), and post-norms with a
        // tighter eps (post_norm_eps = 1e-8) than the pre-norms use.
        const bool is_muse_glimmer = arch_str == "muse-glimmer";
        m_scaleless_embd_norm = is_muse_glimmer;
        m_rope_on_swa_only = is_muse_glimmer;
        m_post_norm_eps = is_muse_glimmer ? 1e-8f : 0.0f;  // 0 -> reuse m_rms_eps

        // ---- qwen35 (Qwen3.5/3.6): hybrid Gated-DeltaNet + full attention ----
        // Layers alternate: every full_attention_interval-th layer is full attention, the rest
        // run a linear-attention (GDN) block. llama.cpp src/models/qwen35.cpp.
        m_is_qwen35 = arch_str == "qwen35";
        m_ssm_conv_kernel = cfg_int("ssm_conv_kernel");
        m_ssm_state_size = cfg_int("ssm_state_size");
        m_ssm_group_count = cfg_int("ssm_group_count");
        m_ssm_dt_rank = cfg_int("ssm_time_step_rank");
        m_ssm_inner_size = cfg_int("ssm_inner_size");
        m_full_attn_interval = cfg_int("full_attention_interval");
        // NextN/MTP blocks live past the main stack and are not executed in a normal forward
        // pass, so the layer loop must stop before them (llama.cpp runs 0..n_layer(), where
        // n_layer() already excludes them; the GGUF block_count counts them in).
        m_n_layer_nextn = cfg_int("nextn_predict_layers");
        if (config.count("recurrent_layer_flags")) {
            if (const auto* v = std::get_if<std::vector<int32_t>>(&config.at("recurrent_layer_flags")))
                m_recurrent_layer_flags = *v;
        }
        if (config.count("rope_sections")) {
            if (const auto* v = std::get_if<std::vector<int32_t>>(&config.at("rope_sections")))
                m_rope_sections = *v;
        }
        if (m_is_qwen35) {
            // blk.0 is a RECURRENT layer whose attn_qkv.weight is the GDN q/k/v/z projection, not
            // a fused attention QKV -- the generic probe above would mistake it for phi-3-style
            // fused QKV and try to split it by attention head counts. Likewise blk.0's
            // attn_gate.weight is the GDN output gate (z), not an attention gate.
            m_has_fused_qkv = false;
            // qwen35's full-attention layers DO have a sigmoid attention output gate, and it is
            // the same construction muse-glimmer uses -- projected from the pre-attention normed
            // hidden, sigmoid'd, multiplied into the merged attention output before wo. The only
            // difference is where the weight lives: qwen35 interleaves it per head inside attn_q
            // rather than storing a separate tensor, so register_qwen35_q_gate() de-interleaves it
            // into blk.N.attn_q.weight + blk.N.attn_gate.weight and the shared path handles it.
            m_has_attn_gate = true;
            // Per-head QK-norm on the full-attention layers. Auto-detection probes blk.0, which is
            // recurrent and carries no attn_q_norm, so set it explicitly.
            m_has_qk_norm = true;
            m_qk_norm_full = false;
            // qwen35's pre-FFN norm key ("post_attention_norm.weight", HF's
            // post_attention_layernorm) already falls out of the generic rule above: attn_norm
            // exists, ffn_norm does not, which is the gpt-oss pattern.
            // MTP/NextN blocks are stored past the main stack and are not part of a normal
            // forward pass, so the layer loop must not walk into them.
            m_n_layer -= m_n_layer_nextn;
            OPENVINO_ASSERT(m_n_layer > 0, "[GGUF] qwen35: no trunk layers left after excluding NextN blocks");
        }

        // Per-architecture scalars from metadata (1.0 / 0.0 when absent -> no-op).
        m_embedding_scale = cfg_float("embedding_scale");
        m_residual_scale = cfg_float("residual_scale");
        m_logit_scale = cfg_float("logit_scale");
        m_attention_scale = cfg_float("attention_scale");            // 0 -> 1/sqrt(head_size)
        m_expert_weights_scale = cfg_float("expert_weights_scale");  // 0 -> 1.0 no-op
        m_rope_freq_base_swa = cfg_float("rope_freq_base_swa");
        m_swa_layer_pattern = cfg_int("swa_layer_pattern");
        // Gemma4: per-layer SWA boolean flags (non-empty when swa_layer_pattern==0).
        if (config.count("swa_layer_flags"))
            m_swa_layer_flags = std::get<std::vector<int32_t>>(config.at("swa_layer_flags"));
        m_attn_soft_cap = cfg_float("attn_logit_softcapping");          // 0 -> no soft-cap
        m_final_logit_soft_cap = cfg_float("final_logit_softcapping");  // 0 -> no soft-cap
        // Gemma4: per-layer input embeddings and shared KV layers.
        m_n_embd_per_layer = cfg_int("n_embd_per_layer");
        m_shared_kv_layers = cfg_int("shared_kv_layers");
        // Gemma4: SWA layers use a smaller head size than global attention layers.
        m_head_size_swa = cfg_int("head_size_swa");
        m_rope_dim_swa = cfg_int("rope_dimension_count_swa");

        const std::string arch = std::get<std::string>(config.at("architecture"));
        // Interleaved M-RoPE (qwen35 / qwen3vl) is its own mode, so it must be decided here rather
        // than in the per-arch block above -- this assignment runs later and would overwrite it.
        m_rope_op_case = m_is_qwen35                 ? ROPE_OP_CASE_IMROPE
                         : arch_uses_neox_rope(arch) ? ROPE_OP_CASE_NEOX
                                                     : ROPE_OP_CASE_NORMAL;

        m_graph->has_rope = true;
        m_graph->rope_config.n_dims = cfg_int("rope_dimension_count");
        m_graph->rope_config.n_ctx_orig = cfg_int("rope_n_ctx_orig");
        m_graph->rope_config.freq_base = cfg_float("rope_freq_base");
        m_graph->rope_config.freq_scale = cfg_float("rope_freq_scale");
        m_graph->rope_config.ext_factor = cfg_float("rope_ext_factor");
        m_graph->rope_config.attn_factor = 1.0f;
        m_graph->rope_config.beta_fast = 32.0f;
        m_graph->rope_config.beta_slow = 1.0f;
        // Gemma4: separate rope config for SWA layers (different freq_base and n_dims).
        m_rope_config_swa = m_graph->rope_config;
        m_rope_config_swa.freq_base = cfg_float("rope_freq_base_swa");
        m_rope_config_swa.n_dims = cfg_int("rope_dimension_count_swa");
        // Per-op sin/cos is required whenever SWA and global layers differ in any rope
        // parameter that feeds the shared sin/cos table: n_dims (gemma4) OR freq_base (gemma3,
        // whose SWA layers rope at 10000 vs the global 1000000). Without it every layer would
        // share the global table and SWA layers would be roped wrong.
        const bool swa_dims_differ = m_rope_dim_swa > 0 && m_rope_dim_swa != m_graph->rope_config.n_dims;
        const bool swa_freq_differs = m_rope_config_swa.freq_base != m_graph->rope_config.freq_base;
        if (m_has_swa && (swa_dims_differ || swa_freq_differs)) {
            m_graph->use_per_op_rope = true;
        }
        // M-RoPE: inp_pos carries 4 sections per token and only the first n_dims of each head
        // are rotated (qwen35: head_size 256, rope.dimension_count 64). Build sin/cos per ROPE op
        // rather than from the shared table, whose layout assumes the single-section full-head
        // case (see RopeConfig::is_imrope / use_per_op_rope).
        if (m_rope_op_case == ROPE_OP_CASE_IMROPE) {
            m_graph->rope_config.is_imrope = true;
            m_graph->use_per_op_rope = true;
        }
    }

    std::shared_ptr<GgufGraph> build();

private:
    int cfg_int(const std::string& k) const {
        auto it = m_config.find(k);
        OPENVINO_ASSERT(it != m_config.end(), "[GGUF] internal: missing config key '", k, "'");
        const auto* v = std::get_if<int>(&it->second);
        OPENVINO_ASSERT(v, "[GGUF] internal: config key '", k, "' is not an int");
        return *v;
    }
    float cfg_float(const std::string& k) const {
        auto it = m_config.find(k);
        OPENVINO_ASSERT(it != m_config.end(), "[GGUF] internal: missing config key '", k, "'");
        const auto* v = std::get_if<float>(&it->second);
        OPENVINO_ASSERT(v, "[GGUF] internal: config key '", k, "' is not a float");
        return *v;
    }

    static ov::PartialShape ps(std::vector<int64_t> dims) {
        return ov::PartialShape(dims);
    }

    // Static shape of an already-emitted tensor, for ops whose translator needs its input's own
    // layout (VIEW op_case 3's "input_ggml_shape", which it uses to restore that layout before
    // slicing). The builder's per-node shapes are static except for the model-input Parameters,
    // so a dynamic dim here means the caller asked about a tensor this cannot describe.
    ov::Shape shape_of(const std::string& tensor_name) const {
        auto it = m_tensor_shapes.find(tensor_name);
        OPENVINO_ASSERT(it != m_tensor_shapes.end(), "[GGUF] internal: no shape recorded for '", tensor_name, "'");
        OPENVINO_ASSERT(it->second.is_static(),
                        "[GGUF] internal: shape of '",
                        tensor_name,
                        "' is dynamic (",
                        it->second,
                        "), cannot be used as a static input shape");
        return it->second.to_shape();
    }

    // Look up a weight tensor by GGUF name, failing with the tensor NAME if the GGUF is missing it
    // (a bare m_weights.at(name) throws std::out_of_range with no context). Used wherever the
    // builder reads a weight's shape to size an op; a missing expected tensor means the file does
    // not match the detected architecture.
    const ov::Tensor& weight_tensor(const std::string& name) const {
        auto it = m_weights.find(name);
        OPENVINO_ASSERT(it != m_weights.end(),
                        "[GGUF] model is missing expected weight tensor '",
                        name,
                        "' for architecture '",
                        std::get<std::string>(m_config.at("architecture")),
                        "'");
        return it->second;
    }

    // Emit a weight as a GGML_OP_NONE leaf node carrying the parser's already-extracted tensors
    // (`<base>.weight` [+ `.scales` [+ `.zp`]] + qtype) as node attributes. translate_weight
    // rebuilds the weights/qtypes maps from these attributes and calls make_weight_node(base,
    // weights, qtypes). Routing weights through GGML_OP_NONE makes that the single weight-loading
    // API, shared with the llama.cpp cgraph decoder (which marks the same leaf with raw ggml bytes
    // instead), and keeps the compressed decompression subgraph -- built lazily in translate_weight
    // during the walk -- rather than materializing an ov::Node eagerly here.
    // `node_name` is the tensor name translators reference (the GGML_OP_NONE output);
    // `base` is that name without the trailing ".weight". `extracted` maps
    // "<base>.weight"/".scales"/".zp" -> tensor; `qtype` is the ggml type of `<base>`.
    void emit_weight_op(const std::string& node_name,
                        const std::unordered_map<std::string, ov::Tensor>& extracted,
                        gguf_tensor_type qtype,
                        const ov::PartialShape& shape_4d) {
        if (m_emitted_weights.count(node_name)) {
            return;
        }
        m_emitted_weights.insert(node_name);

        GgufOp op;
        op.op_type = "GGML_OP_NONE";
        op.name = node_name;
        op.output_name = node_name;
        op.output_shape = shape_4d;
        op.output_type = ov::element::f32;
        // translate_weight rebuilds the make_weight_node(base, weights, qtypes) inputs from these:
        // "gguf.blob.<sub>" -> extracted tensor (<sub> in {weight, scales, zp}), and the qtype id.
        op.attributes["gguf_weight"] = true;  // marks this GGML_OP_NONE leaf as a weight
        op.attributes["gguf_qtype"] = static_cast<int>(qtype);
        for (const auto& kv : extracted) {
            const std::string& full = kv.first;  // "<base>.weight" / ".scales" / ".zp"
            auto dot = full.rfind('.');
            std::string sub = (dot == std::string::npos) ? full : full.substr(dot + 1);
            op.attributes["gguf.blob." + sub] = kv.second;
        }
        m_graph->nodes.push_back(std::move(op));

        m_tensor_shapes[node_name] = shape_4d;
        m_tensor_types[node_name] = ov::element::f32;
    }

    // `ggml_name` is the full tensor name ending in ".weight" (the name translators reference).
    void add_weight(const std::string& ggml_name) {
        if (m_emitted_weights.count(ggml_name)) {
            return;
        }
        const std::string suffix = ".weight";
        const std::string base = (ggml_name.size() > suffix.size() &&
                                  ggml_name.compare(ggml_name.size() - suffix.size(), suffix.size(), suffix) == 0)
                                     ? ggml_name.substr(0, ggml_name.size() - suffix.size())
                                     : ggml_name;

        // Collect the parser's extracted tensors for this weight (weight [+ scales [+ zp]]).
        std::unordered_map<std::string, ov::Tensor> extracted;
        for (const char* sub : {".weight", ".scales", ".zp"}) {
            auto it = m_weights.find(base + sub);
            if (it != m_weights.end()) {
                extracted[base + sub] = it->second;
            }
        }
        gguf_tensor_type qtype = GGUF_TYPE_F16;
        if (auto it = m_qtypes.find(base + ".qtype"); it != m_qtypes.end()) {
            qtype = it->second;
        }

        // Shape padded to the decoder's 4D reversed layout [1, 1, rows, cols] so add_op can fill
        // per-input shape/type for translators (MUL_MAT) that index dims [1] and [3].
        int64_t rows = 1, cols = 1;
        if (auto it = m_weights.find(ggml_name); it != m_weights.end()) {
            const auto& s = it->second.get_shape();  // [rows, cols(packed)]
            rows = s.size() >= 1 ? static_cast<int64_t>(s[0]) : 1;
            cols = s.size() >= 2 ? static_cast<int64_t>(s[1]) : 1;
        }
        emit_weight_op(ggml_name, extracted, qtype, ov::PartialShape({1, 1, rows, cols}));
    }

    // Emit a weight node `node_name` (ends in ".weight") reusing the parser's extracted tensors of
    // another weight `src_base` (base without ".weight"). Used for MQA tie-V, where V shares K's
    // weight tensor; the two GGML_OP_NONE leaves reference the same underlying ov::Tensor blobs
    // (cheap: SharedBuffer views into the parser's single quant buffer).
    void add_weight_from(const std::string& node_name, const std::string& src_base) {
        if (m_emitted_weights.count(node_name)) {
            return;
        }
        const std::string suffix = ".weight";
        const std::string dst_base = (node_name.size() > suffix.size() &&
                                      node_name.compare(node_name.size() - suffix.size(), suffix.size(), suffix) == 0)
                                         ? node_name.substr(0, node_name.size() - suffix.size())
                                         : node_name;
        std::unordered_map<std::string, ov::Tensor> extracted;
        for (const char* sub : {".weight", ".scales", ".zp"}) {
            auto it = m_weights.find(src_base + sub);
            if (it != m_weights.end()) {
                extracted[dst_base + sub] = it->second;
            }
        }
        gguf_tensor_type qtype = GGUF_TYPE_F16;
        if (auto it = m_qtypes.find(src_base + ".qtype"); it != m_qtypes.end()) {
            qtype = it->second;
        }
        int64_t rows = 1, cols = 1;
        if (auto it = m_weights.find(src_base + ".weight"); it != m_weights.end()) {
            const auto& s = it->second.get_shape();
            rows = s.size() >= 1 ? static_cast<int64_t>(s[0]) : 1;
            cols = s.size() >= 2 ? static_cast<int64_t>(s[1]) : 1;
        }
        emit_weight_op(node_name, extracted, qtype, ov::PartialShape({1, 1, rows, cols}));
    }

    std::shared_ptr<ov::op::v0::Parameter> add_input(const std::string& name,
                                                     ov::element::Type type,
                                                     const ov::PartialShape& shape) {
        auto p = std::make_shared<ov::op::v0::Parameter>(type, shape);
        p->set_friendly_name(name);
        p->output(0).set_names({name});
        m_graph->model_inputs[name] = p;
        return p;
    }

    void add_extra_input(const std::string& name, int64_t value) {
        auto c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {value});
        c->set_friendly_name(name);
        m_graph->model_extra_inputs[name] = c;
    }

    // Append one op node. `inputs` are producer tensor names (weights / model inputs /
    // earlier node outputs). Returns the output tensor name (== node name).
    std::string add_op(const std::string& op_type,
                       const std::string& name,
                       const std::vector<std::string>& inputs,
                       const ov::PartialShape& out_shape,
                       ov::element::Type out_type,
                       int op_case = 0,
                       std::map<std::string, ov::Any> attrs = {}) {
        GgufOp op;
        op.op_type = op_type;
        op.name = name;
        op.input_names = inputs;
        op.output_name = name;
        op.output_shape = out_shape;
        op.output_type = out_type;
        op.op_case = op_case;
        op.attributes = std::move(attrs);
        // Fill per-input shape/type from known producers so translators that query them
        // (MUL_MAT, RESHAPE) get sane values.
        for (const auto& in : inputs) {
            if (auto it = m_tensor_shapes.find(in); it != m_tensor_shapes.end()) {
                op.input_shapes[in] = it->second;
            }
            if (auto it = m_tensor_types.find(in); it != m_tensor_types.end()) {
                op.input_types[in] = it->second;
            }
        }
        m_tensor_shapes[name] = out_shape;
        m_tensor_types[name] = out_type;
        m_graph->nodes.push_back(std::move(op));
        return name;
    }

    // RMS_NORM followed by elementwise MUL with the norm weight (build_norm, LLM_NORM_RMS).
    // `eps` overrides the model-wide rms_norm_eps when > 0 (muse-glimmer's post-norms use a
    // tighter 1e-8 than its pre-norms).
    std::string rms_norm(const std::string& in,
                         const std::string& weight,
                         const std::string& out_prefix,
                         float eps = 0.0f) {
        add_weight(weight);
        auto norm = add_op("GGML_OP_RMS_NORM",
                           out_prefix + ".rms",
                           {in},
                           m_tensor_shapes.at(in),
                           ov::element::f32,
                           0,
                           {{"eps", eps > 0.0f ? eps : m_rms_eps}});
        m_tensor_shapes[weight] = m_tensor_shapes.count(weight) ? m_tensor_shapes[weight] : ov::PartialShape{};
        return add_op("GGML_OP_MUL", out_prefix, {norm, weight}, m_tensor_shapes.at(in), ov::element::f32);
    }

    // Dense GeGLU FFN (Gemma/Gemma2). Same layout as SwiGLU but uses GELU activation.
    // `ffn_norm` is the post-norm hidden. Returns the down-projection output tensor name.
    std::string build_geglu_ffn(const std::string& p, const std::string& ffn_norm, int64_t T) {
        using ov::element::f32;
        add_weight(p + "ffn_gate.weight");
        add_weight(p + "ffn_up.weight");
        add_weight(p + "ffn_down.weight");
        const int n_ff = static_cast<int>(weight_tensor(p + "ffn_gate.weight").get_shape()[0]);
        auto gate =
            add_op("GGML_OP_MUL_MAT", p + "ffn_gate", {p + "ffn_gate.weight", ffn_norm}, ps({1, 1, T, n_ff}), f32);
        auto up = add_op("GGML_OP_MUL_MAT", p + "ffn_up", {p + "ffn_up.weight", ffn_norm}, ps({1, 1, T, n_ff}), f32);
        auto glu =
            add_op("GGML_GLU_OP_GEGLU", p + "ffn_geglu", {gate, up}, ps({1, 1, T, n_ff}), f32, 0, {{"swapped", false}});
        return add_op("GGML_OP_MUL_MAT", p + "ffn_out", {p + "ffn_down.weight", glu}, ps({1, 1, T, m_n_embd}), f32);
    }

    // Dense SwiGLU FFN (llama/qwen/phi3/minicpm). `ffn_norm` is the post-norm hidden.
    // Returns the down-projection output tensor name. T is the static token length.
    // Supports optional per-projection biases (pangu-embedded: ffn_gate.bias, ffn_up.bias,
    // ffn_down.bias), auto-detected from the weight table at layer 0.
    std::string build_dense_ffn(const std::string& p, const std::string& ffn_norm, int64_t T) {
        using ov::element::f32;
        add_weight(p + "ffn_up.weight");
        add_weight(p + "ffn_down.weight");
        const bool fused_ffn = m_weights.count(p + "ffn_gate.weight") == 0;  // phi-3: fused gate+up
        const bool has_ffn_bias = m_weights.count(p + "ffn_up.bias") > 0;
        std::string glu;
        if (fused_ffn) {
            const int up2 = static_cast<int>(weight_tensor(p + "ffn_up.weight").get_shape()[0]);  // 2*n_ff
            auto up = add_op("GGML_OP_MUL_MAT", p + "ffn_up", {p + "ffn_up.weight", ffn_norm}, ps({1, 1, T, up2}), f32);
            glu = add_op("GGML_GLU_OP_SWIGLU",
                         p + "ffn_swiglu",
                         {up},
                         ps({1, 1, T, up2 / 2}),
                         f32,
                         0,
                         {{"swapped", false}});
        } else {
            add_weight(p + "ffn_gate.weight");
            const int n_ff = static_cast<int>(weight_tensor(p + "ffn_gate.weight").get_shape()[0]);
            auto gate =
                add_op("GGML_OP_MUL_MAT", p + "ffn_gate", {p + "ffn_gate.weight", ffn_norm}, ps({1, 1, T, n_ff}), f32);
            if (has_ffn_bias) {
                gate = add_bias(gate, p + "ffn_gate.bias", p + "ffn_gate_b");
            }
            auto up =
                add_op("GGML_OP_MUL_MAT", p + "ffn_up", {p + "ffn_up.weight", ffn_norm}, ps({1, 1, T, n_ff}), f32);
            if (has_ffn_bias) {
                up = add_bias(up, p + "ffn_up.bias", p + "ffn_up_b");
            }
            glu = add_op("GGML_GLU_OP_SWIGLU",
                         p + "ffn_swiglu",
                         {gate, up},
                         ps({1, 1, T, n_ff}),
                         f32,
                         0,
                         {{"swapped", false}});
        }
        auto out = add_op("GGML_OP_MUL_MAT", p + "ffn_out", {p + "ffn_down.weight", glu}, ps({1, 1, T, m_n_embd}), f32);
        if (has_ffn_bias) {
            out = add_bias(out, p + "ffn_down.bias", p + "ffn_out_b");
        }
        return out;
    }

    // Mixture-of-experts FFN (OLMoE / gpt-oss), mirroring llm_graph_context::build_moe_ffn.
    // Routing: logits = gate_inp·x; probs = softmax/identity; pick top-k experts; per-token
    // expert matmuls via MUL_MAT_ID; gated activation; weighted sum over the used experts.
    std::string build_moe_ffn(const std::string& p, const std::string& ffn_norm, int64_t T) {
        using ov::element::f32;
        const int E = m_n_expert, K = m_n_expert_used;
        const int n_ff = static_cast<int>(weight_tensor(p + "ffn_gate_exps.weight").get_shape()[1]);

        // --- router: logits [1,1,T,E] = gate_inp · x ---
        add_weight(p + "ffn_gate_inp.weight");
        auto logits =
            add_op("GGML_OP_MUL_MAT", p + "moe_logits", {p + "ffn_gate_inp.weight", ffn_norm}, ps({1, 1, T, E}), f32);
        if (m_has_moe_gate_bias) {
            logits = add_bias(logits, p + "ffn_gate_inp.bias", p + "moe_logits_b");
        }

        // gating: softmax (OLMoE) or softmax-after-topk (gpt-oss "softmax_weight").
        std::string probs = logits;
        if (!m_moe_softmax_weight) {
            probs = add_op("GGML_OP_SOFT_MAX",
                           p + "moe_probs",
                           {logits},
                           ps({1, 1, T, E}),
                           f32,
                           0,
                           {{"softmax_axis", int64_t(-1)}});
        }

        // top-k expert selection -> indices [1,1,T,K] (i32).
        auto selected = add_op("GGML_OP_TOP_K", p + "moe_topk", {probs}, ps({1, 1, T, K}), ov::element::i32);

        // weights = gather probs by selected; op_case 10 returns a per-expert column
        // [1,T,K,1] (robust to dynamic T). gpt-oss softmaxes over the K (expert) axis.
        auto weights = add_op("GGML_OP_GET_ROWS", p + "moe_w", {probs, selected}, ps({1, T, K, 1}), f32, 10);
        if (m_moe_softmax_weight) {
            weights = add_op("GGML_OP_SOFT_MAX",
                             p + "moe_w_sm",
                             {weights},
                             ps({1, T, K, 1}),
                             f32,
                             0,
                             {{"softmax_axis", int64_t(2)}});
        }
        // gpt-oss expert_weights_scale: optional constant multiplier applied after softmax
        // (mirrors llama.cpp build_moe_ffn w_scale != 0 && w_scale != 1.0 path).
        if (m_expert_weights_scale != 0.0f && m_expert_weights_scale != 1.0f) {
            weights = scale(weights, m_expert_weights_scale, p + "moe_w_scaled");
        }

        // expert FFN via MUL_MAT_ID. The routed input x is broadcast to K slots; the
        // translator gathers each token's selected expert matrices. gpt-oss adds per-expert
        // biases (ADD_ID gathers the selected experts' bias rows).
        add_weight(p + "ffn_gate_exps.weight");
        add_weight(p + "ffn_up_exps.weight");
        add_weight(p + "ffn_down_exps.weight");
        const bool eb = m_has_moe_expert_bias;
        auto up = add_op("GGML_OP_MUL_MAT_ID",
                         p + "moe_up",
                         {p + "ffn_up_exps.weight", ffn_norm, selected},
                         ps({1, T, K, n_ff}),
                         f32);
        if (eb) {
            add_named_weight(p + "ffn_up_exps.bias");
            up = add_op("GGML_OP_ADD_ID",
                        p + "moe_up_b",
                        {up, p + "ffn_up_exps.bias", selected},
                        ps({1, T, K, n_ff}),
                        f32);
        }
        auto gate = add_op("GGML_OP_MUL_MAT_ID",
                           p + "moe_gate",
                           {p + "ffn_gate_exps.weight", ffn_norm, selected},
                           ps({1, T, K, n_ff}),
                           f32);
        if (eb) {
            add_named_weight(p + "ffn_gate_exps.bias");
            gate = add_op("GGML_OP_ADD_ID",
                          p + "moe_gate_b",
                          {gate, p + "ffn_gate_exps.bias", selected},
                          ps({1, T, K, n_ff}),
                          f32);
        }
        std::string act;
        if (m_moe_swiglu_oai) {
            act = add_op("GGML_GLU_OP_SWIGLU_OAI",
                         p + "moe_act",
                         {gate, up},
                         ps({1, T, K, n_ff}),
                         f32,
                         0,
                         // Attribute names must match the cgraph decoder's (ggml-decoder.cpp), which
                         // is the external contract the shared translators read: glu_alpha/glu_limit.
                         {{"swapped", false}, {"glu_alpha", 1.702f}, {"glu_limit", 7.0f}});
        } else {
            act = add_op("GGML_GLU_OP_SWIGLU",
                         p + "moe_act",
                         {gate, up},
                         ps({1, T, K, n_ff}),
                         f32,
                         0,
                         {{"swapped", false}});
        }
        auto experts = add_op("GGML_OP_MUL_MAT_ID",
                              p + "moe_down",
                              {p + "ffn_down_exps.weight", act, selected},
                              ps({1, T, K, m_n_embd}),
                              f32);
        if (eb) {
            add_named_weight(p + "ffn_down_exps.bias");
            experts = add_op("GGML_OP_ADD_ID",
                             p + "moe_down_b",
                             {experts, p + "ffn_down_exps.bias", selected},
                             ps({1, T, K, m_n_embd}),
                             f32);
        }

        // Weighted sum over the K selected experts. weights is [1,T,K,1] (per-expert col).
        //   experts [1,T,K,n_embd] * weights -> [1,T,K,n_embd]
        //   TRANSPOSE last two axes -> [1,T,n_embd,K]; SUM_ROWS over K -> [1,T,n_embd,1]
        //   RESHAPE (dynamic) -> [1,1,T,n_embd].
        // The reshape is op_case 5: collapse to [1, 1, -1, n_embd], the token axis staying dynamic.
        // That is the case the cgraph decoder would assign to this same reshape too -- ggml ne
        // [1,n_embd,T,1] -> [n_embd,T,1,1] satisfies both of its case-5 predicates -- so the shared
        // case applies as-is and needs no builder-specific numbering.
        auto weighted = add_op("GGML_OP_MUL", p + "moe_weighted", {experts, weights}, ps({1, T, K, m_n_embd}), f32);
        auto tr = add_op("GGML_OP_TRANSPOSE", p + "moe_tr", {weighted}, ps({1, T, m_n_embd, K}), f32);
        auto summed = add_op("GGML_OP_SUM_ROWS", p + "moe_sum", {tr}, ps({1, T, m_n_embd, 1}), f32);
        auto moe_out = add_op("GGML_OP_RESHAPE", p + "moe_out", {summed}, ps({1, 1, T, m_n_embd}), f32, 5);

        // Shared experts (deepseek2-ocr, bailingmoe2, exaone-moe): always-active experts whose
        // output is added to the routed experts' weighted sum. Uses plain SwiGLU dense FFN with
        // ffn_{gate,up,down}_shexp.weight (n_ff_shared = shexp rows). Output added to moe_out.
        if (m_n_expert_shared > 0 && m_weights.count(p + "ffn_gate_shexp.weight") > 0) {
            add_weight(p + "ffn_gate_shexp.weight");
            add_weight(p + "ffn_up_shexp.weight");
            add_weight(p + "ffn_down_shexp.weight");
            const int n_ff_s = static_cast<int>(weight_tensor(p + "ffn_gate_shexp.weight").get_shape()[0]);
            auto s_gate = add_op("GGML_OP_MUL_MAT",
                                 p + "shexp_gate",
                                 {p + "ffn_gate_shexp.weight", ffn_norm},
                                 ps({1, 1, T, n_ff_s}),
                                 f32);
            auto s_up = add_op("GGML_OP_MUL_MAT",
                               p + "shexp_up",
                               {p + "ffn_up_shexp.weight", ffn_norm},
                               ps({1, 1, T, n_ff_s}),
                               f32);
            auto s_act = add_op("GGML_GLU_OP_SWIGLU",
                                p + "shexp_act",
                                {s_gate, s_up},
                                ps({1, 1, T, n_ff_s}),
                                f32,
                                0,
                                {{"swapped", false}});
            auto s_down = add_op("GGML_OP_MUL_MAT",
                                 p + "shexp_out",
                                 {p + "ffn_down_shexp.weight", s_act},
                                 ps({1, 1, T, m_n_embd}),
                                 f32);
            moe_out = add_op("GGML_OP_ADD", p + "moe_shared_out", {moe_out, s_down}, ps({1, 1, T, m_n_embd}), f32);
        }
        return moe_out;
    }

    // Split a fused blk.<il>.attn_qkv weight into separate q/k/v weight nodes registered
    // under blk.<il>.attn_{q,k,v}.weight, so the rest of the builder is identical to the
    // separate-projection case. Rows split as [n_head*head_size | n_head_kv*head_size x2].
    void register_fused_qkv(int il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const size_t n_q = static_cast<size_t>(m_n_head) * m_head_size;
        const size_t n_kv = static_cast<size_t>(m_n_head_kv) * m_head_size;
        auto parts = split_fused_qkv_extracted(p + "attn_qkv", m_weights, m_qtypes, n_q, n_kv, n_kv);
        const std::array<std::string, 3> names = {p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"};
        const std::array<int64_t, 3> rows = {(int64_t)n_q, (int64_t)n_kv, (int64_t)n_kv};
        for (size_t i = 0; i < 3; ++i) {
            // Re-key the sliced tensors from "<base>.q.weight" to "<node>.weight" so emit_weight_op's
            // sub-key extraction (weight/scales/zp) and translate_weight's make_weight_node see the
            // node's own base name.
            std::unordered_map<std::string, ov::Tensor> extracted;
            const std::string base = p + "attn_" + std::string(1, "qkv"[i]);  // blk.N.attn_q etc.
            for (const auto& kv : parts[i].extracted) {
                auto dot = kv.first.rfind('.');
                extracted[base + "." + kv.first.substr(dot + 1)] = kv.second;
            }
            emit_weight_op(names[i], extracted, parts[i].qtype, ps({1, 1, rows[i], m_n_embd}));
        }
    }

    // qwen35 full-attention layers: attn_q packs query and attention-output gate interleaved per
    // head ([q_h0 | gate_h0 | q_h1 | ...], stride 2*head_size). De-interleave into the plain
    // blk.N.attn_q.weight / blk.N.attn_gate.weight the shared attention path expects, so the
    // graph never sees the interleaving. llama.cpp does the same split with two strided views
    // (src/models/qwen35.cpp build_layer_attn).
    void register_qwen35_q_gate(int il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        auto parts = split_interleaved_q_gate(p + "attn_q", m_weights, m_qtypes, static_cast<size_t>(m_head_size));
        const std::array<std::string, 2> names = {p + "attn_q.weight", p + "attn_gate.weight"};
        const std::array<std::string, 2> suffix = {"q", "gate"};
        const int64_t rows = static_cast<int64_t>(m_n_head) * m_head_size;
        for (size_t i = 0; i < 2; ++i) {
            // Re-key "<base>.q.weight" -> "<node>.weight" so emit_weight_op's sub-key extraction
            // sees the node's own base name (same re-keying as register_fused_qkv).
            std::unordered_map<std::string, ov::Tensor> extracted;
            const std::string base = p + "attn_" + suffix[i];
            for (const auto& kv : parts[i].extracted) {
                auto dot = kv.first.rfind('.');
                extracted[base + "." + kv.first.substr(dot + 1)] = kv.second;
            }
            emit_weight_op(names[i], extracted, parts[i].qtype, ps({1, 1, rows, m_n_embd}));
        }
    }

    // ---- qwen35 linear-attention (Gated DeltaNet) layer ----
    // Mirrors llama.cpp src/models/qwen35.cpp build_layer_attn_linear + delta-net-base.cpp.
    // `attn_norm` is the pre-attention normed hidden; returns the sublayer output BEFORE the
    // residual add, i.e. exactly what the shared tail expects in place of attn_out.
    //
    // The two recurrent states are plain model Parameters written through to Results, matching
    // how the builder treats KV caches: the frontend always emits a STATELESS graph and leaves
    // statefulness to the consumer. Unlike a KV cache these are OVERWRITTEN, not appended, so
    // they carry no token axis and MakeStateful's Concat path does not apply to them.
    std::string build_qwen35_gdn(int il, const std::string& attn_norm, int64_t T) {
        using ov::element::f32;
        const std::string p = "blk." + std::to_string(il) + ".";

        const int64_t d_conv = m_ssm_conv_kernel;
        const int64_t S = m_ssm_state_size;             // head_k_dim == head_v_dim
        const int64_t H_k = m_ssm_group_count;          // num_k_heads
        const int64_t H_v = m_ssm_dt_rank;              // num_v_heads
        const int64_t head_v = m_ssm_inner_size / H_v;  // head_v_dim
        const int64_t key_dim = S * H_k;
        const int64_t value_dim = head_v * H_v;
        const int64_t conv_dim = 2 * key_dim + value_dim;

        // ---- input projections ----
        add_weight(p + "attn_qkv.weight");
        auto qkv = add_op("GGML_OP_MUL_MAT",
                          p + "qkv_mixed",
                          {p + "attn_qkv.weight", attn_norm},
                          ps({1, 1, T, conv_dim}),
                          f32);
        add_weight(p + "attn_gate.weight");
        auto z = add_op("GGML_OP_MUL_MAT", p + "z", {p + "attn_gate.weight", attn_norm}, ps({1, 1, T, value_dim}), f32);

        // beta = sigmoid(ssm_beta @ x), one scalar per v-head
        add_weight(p + "ssm_beta.weight");
        auto beta = add_op("GGML_OP_MUL_MAT", p + "beta", {p + "ssm_beta.weight", attn_norm}, ps({1, 1, T, H_v}), f32);
        beta = add_op("GGML_UNARY_OP_SIGMOID", p + "beta_sig", {beta}, ps({1, 1, T, H_v}), f32);
        beta = add_op("GGML_OP_RESHAPE", p + "beta_4d", {beta}, ps({1, T, H_v, 1}), f32, 1);

        // g = softplus(ssm_alpha @ x + ssm_dt.bias) * ssm_a   (ggml: -A_log.exp() * softplus)
        add_weight(p + "ssm_alpha.weight");
        auto alpha =
            add_op("GGML_OP_MUL_MAT", p + "alpha", {p + "ssm_alpha.weight", attn_norm}, ps({1, 1, T, H_v}), f32);
        add_named_weight(p + "ssm_dt.bias");
        alpha = add_op("GGML_OP_ADD", p + "alpha_biased", {alpha, p + "ssm_dt.bias"}, ps({1, 1, T, H_v}), f32);
        alpha = add_op("GGML_UNARY_OP_SOFTPLUS", p + "alpha_sp", {alpha}, ps({1, 1, T, H_v}), f32);
        add_named_weight(p + "ssm_a");
        auto g = add_op("GGML_OP_MUL", p + "gate", {alpha, p + "ssm_a"}, ps({1, 1, T, H_v}), f32);
        g = add_op("GGML_OP_RESHAPE", p + "gate_4d", {g}, ps({1, T, H_v, 1}), f32, 1);

        // ---- causal depthwise conv over [conv state | this step's tokens] ----
        // conv_state holds the trailing d_conv-1 columns of the previous step's conv input.
        const std::string cs = "conv_state_l" + std::to_string(il);
        if (!m_graph->model_inputs.count(cs)) {
            add_input(cs, f32, ps({1, 1, conv_dim, d_conv - 1}));
        }
        m_tensor_shapes[cs] = ps({1, 1, conv_dim, d_conv - 1});
        m_tensor_types[cs] = f32;

        // [1,1,T,conv_dim] -> [1,1,conv_dim,T] so the conv window grows along the last axis.
        auto qkv_t = add_op("GGML_OP_TRANSPOSE", p + "qkv_t", {qkv}, ps({1, 1, conv_dim, T}), f32);
        auto conv_in = add_op("GGML_OP_CONCAT",
                              p + "conv_in",
                              {cs, qkv_t},
                              ps({1, 1, conv_dim, d_conv - 1 + T}),
                              f32,
                              0,
                              {{"concat_axis", int{0}}});

        // Next step's state is the trailing d_conv-1 columns of this window.
        const std::vector<int64_t> tail_slice{3, -(d_conv - 1), d_conv - 1};
        auto cs_out = add_op("GGML_OP_VIEW",
                             cs + "_out",
                             {conv_in},
                             ps({1, 1, conv_dim, d_conv - 1}),
                             f32,
                             3,
                             {{"view_slice", tail_slice}, {"input_ggml_shape", shape_of(conv_in)}});
        m_graph->model_output_names.push_back(cs_out);
        m_graph->recurrent_states.emplace_back(cs, cs_out);

        add_named_weight(p + "ssm_conv1d.weight");
        auto conv = add_op("GGML_OP_SSM_CONV",
                           p + "conv_out",
                           {conv_in, p + "ssm_conv1d.weight"},
                           ps({1, 1, T, conv_dim}),
                           f32);
        conv = add_op("GGML_UNARY_OP_SILU", p + "conv_silu", {conv}, ps({1, 1, T, conv_dim}), f32);

        // ---- split the conv output into q | k | v and normalize q/k ----
        auto slice_heads = [&](const std::string& name, int64_t off, int64_t width, int64_t heads, int64_t dim) {
            const std::vector<int64_t> sl{3, off, width};
            auto s = add_op("GGML_OP_VIEW",
                            p + name + "_s",
                            {conv},
                            ps({1, 1, T, width}),
                            f32,
                            3,
                            {{"view_slice", sl}, {"input_ggml_shape", shape_of(conv)}});
            return add_op("GGML_OP_RESHAPE", p + name, {s}, ps({1, T, heads, dim}), f32, 1);
        };
        auto q = slice_heads("q_conv", 0, key_dim, H_k, S);
        auto k = slice_heads("k_conv", key_dim, key_dim, H_k, S);
        auto v = slice_heads("v_conv", 2 * key_dim, value_dim, H_v, head_v);
        q = add_op("GGML_OP_L2_NORM", p + "q_l2", {q}, ps({1, T, H_k, S}), f32, 0, {{"eps", m_rms_eps}});
        k = add_op("GGML_OP_L2_NORM", p + "k_l2", {k}, ps({1, T, H_k, S}), f32, 0, {{"eps", m_rms_eps}});

        // ---- recurrent delta rule ----
        // ggml state layout is [B, H_v, value_dim, key_dim]; the translator transposes it for
        // the fused op and transposes the new state back, so the Parameter keeps ggml's layout.
        const std::string ss = "ssm_state_l" + std::to_string(il);
        if (!m_graph->model_inputs.count(ss)) {
            add_input(ss, f32, ps({1, H_v, head_v, S}));
        }
        m_tensor_shapes[ss] = ps({1, H_v, head_v, S});
        m_tensor_types[ss] = f32;

        auto gdn = add_op("GGML_OP_GATED_DELTA_NET",
                          p + "gdn",
                          {q, k, v, g, beta, ss},
                          ps({1, 1, T + head_v, head_v * H_v}),
                          f32,
                          0,
                          {{"gdn_state_slots", int64_t{1}}});

        // The op packs [attn rows | new-state rows]; op_case 4 slices them apart. The attn view's
        // token axis is marked -1 explicitly rather than left as the representative T: the consumer
        // replaces the FIRST dim equal to the token count, and with T == 1 that would match the
        // leading batch dim and move the tokens into dim 0 for the rest of the model.
        const std::vector<int64_t> attn_view{0, head_v};
        const std::vector<int64_t> state_view{1, head_v};
        auto attn = add_op("GGML_OP_VIEW",
                           p + "gdn_attn",
                           {gdn},
                           ps({1, T, H_v, head_v}),
                           f32,
                           4,
                           {{"gdn_view", attn_view}, {"view_reshape", std::vector<int64_t>{1, -1, H_v, head_v}}});
        auto new_state = add_op("GGML_OP_VIEW",
                                ss + "_out",
                                {gdn},
                                ps({1, H_v, head_v, S}),
                                f32,
                                4,
                                {{"gdn_view", state_view}, {"view_reshape", std::vector<int64_t>{1, H_v, head_v, S}}});
        m_graph->model_output_names.push_back(new_state);
        m_graph->recurrent_states.emplace_back(ss, new_state);

        // ---- gated output norm + projection ----
        // build_norm_gated: rms_norm(attn, ssm_norm) * silu(z), normalizing the head_v axis.
        auto out = rms_norm(attn, p + "ssm_norm.weight", p + "gdn_norm");
        auto z_4d = add_op("GGML_OP_RESHAPE", p + "z_4d", {z}, ps({1, T, H_v, head_v}), f32, 1);
        auto z_silu = add_op("GGML_UNARY_OP_SILU", p + "z_silu", {z_4d}, ps({1, T, H_v, head_v}), f32);
        out = add_op("GGML_OP_MUL", p + "gdn_gated", {out, z_silu}, ps({1, T, H_v, head_v}), f32);
        out = add_op("GGML_OP_RESHAPE", p + "gdn_merged", {out}, ps({1, 1, T, value_dim}), f32, 2);

        add_weight(p + "ssm_out.weight");
        return add_op("GGML_OP_MUL_MAT",
                      p + "linear_attn_out",
                      {p + "ssm_out.weight", out},
                      ps({1, 1, T, m_n_embd}),
                      f32);
    }

    // Emit a plain (non-quantized) weight stored under its full GGUF name, e.g. a bias tensor
    // "blk.N.attn_q.bias" (no ".weight" suffix). It flows through the same GGML_OP_NONE +
    // translate_weight path; make_weight_node treats an F16/F32/BF16 blob as a plain Constant.
    void add_named_weight(const std::string& ggml_name) {
        if (m_emitted_weights.count(ggml_name)) {
            return;
        }
        auto it = m_weights.find(ggml_name);
        OPENVINO_ASSERT(it != m_weights.end(), "[GGUF] weight not found in gguf: ", ggml_name);
        const ov::Tensor& w = it->second;
        // Map the OV element type back to the ggml float qtype so translate_weight rebuilds a plain
        // Constant of the right precision.
        gguf_tensor_type qtype = w.get_element_type() == ov::element::f32    ? GGUF_TYPE_F32
                                 : w.get_element_type() == ov::element::bf16 ? GGUF_TYPE_BF16
                                                                             : GGUF_TYPE_F16;
        // emit_weight_op re-keys by the last '.'; a bias ends in ".bias", so key it as ".weight"
        // explicitly (make_weight_node's plain-Constant path reads "<base>.weight").
        std::unordered_map<std::string, ov::Tensor> extracted{{ggml_name + ".weight", w}};
        const auto& s = w.get_shape();
        int64_t n = s.empty() ? 1 : static_cast<int64_t>(s[0]);
        // 2-D plain weights (qwen35's ssm_conv1d, OV [conv_dim, d_conv]) keep both extents;
        // 1-D ones (biases, norm scales) are the common case and stay a trailing vector.
        const ov::PartialShape shape = s.size() == 2 ? ps({1, 1, n, static_cast<int64_t>(s[1])}) : ps({1, 1, 1, n});
        emit_weight_op(ggml_name, extracted, qtype, shape);
    }

    // Elementwise add of a (broadcast) bias weight: GGML_OP_ADD(x, bias_weight).
    std::string add_bias(const std::string& x, const std::string& bias_weight, const std::string& name) {
        add_named_weight(bias_weight);
        return add_op("GGML_OP_ADD", name, {x, bias_weight}, m_tensor_shapes.at(x), ov::element::f32);
    }

    // Scale a tensor by a constant: GGML_OP_SCALE with attr "scale" (and bias 0).
    std::string scale(const std::string& x, float factor, const std::string& name) {
        return add_op("GGML_OP_SCALE",
                      name,
                      {x},
                      m_tensor_shapes.at(x),
                      ov::element::f32,
                      0,
                      {{"scale", factor}, {"bias", 0.0f}});
    }

    // Turn a flat per-token per-layer embedding [1, 1, T, n_layer * n_embd_per_layer] into the
    // layer-major [1, n_layer, T, n_embd_per_layer] the per-layer VIEW slices from.
    //
    // The data is contiguous as [T, n_layer, pe] (one row per token), so this is a reshape that
    // splits the last axis followed by a transpose of the two middle axes -- a single reshape
    // straight to the target would be wrong for T > 1. Emitted as the same two steps ggml uses
    // (ggml_reshape_3d, then ggml_cont(ggml_permute)), so both reach the shared op cases: RESHAPE
    // case 1 splits the last dim into [.., n_layer, pe] and PERMUTE case 1 is the {0,2,1,3}
    // transpose. Fusing them into one node would need a builder-only case for the transpose. No
    // CONT node: translate_cont's PERMUTE case is a pass-through, since translate_permute already
    // emitted a real Transpose, so ggml's ggml_cont would convert to nothing here.
    std::string reshape_to_layer_major(const std::string& flat, const std::string& name) {
        const auto& flat_shape = m_tensor_shapes.at(flat);
        const int64_t T = flat_shape[2].is_static() ? flat_shape[2].get_length() : 1;
        const auto f32 = ov::element::f32;
        auto split =
            add_op("GGML_OP_RESHAPE", name + "_split", {flat}, ps({1, T, m_n_layer, m_n_embd_per_layer}), f32, 1);
        return add_op("GGML_OP_PERMUTE", name, {split}, ps({1, m_n_layer, T, m_n_embd_per_layer}), f32, 1);
    }

    // ---- Per-layer hyperparameter accessors ----
    // Centralize the per-layer derivations that vary by architecture (SWA layers, variable head
    // counts). Previously these were recomputed inline (and duplicated) inside build(); keeping
    // them here is the single source of truth and makes the main loop read declaratively.

    // Whether attention layer `il` uses a sliding window. gemma4 carries an explicit per-layer
    // boolean array; gpt-oss/gemma3/smollm3 use a period (il is SWA unless it is the last in each
    // period, matching llama.cpp set_swa_pattern(period, dense_first=false)).
    bool layer_is_swa(int il) const {
        if (!m_swa_layer_flags.empty()) {
            return il < static_cast<int>(m_swa_layer_flags.size()) && m_swa_layer_flags[il] != 0;
        }
        return m_has_swa && m_swa_layer_pattern > 0 && (il % m_swa_layer_pattern < (m_swa_layer_pattern - 1));
    }

    // Head size for layer `il`. gemma4 SWA layers use a smaller head size than global layers.
    int head_size(int il) const {
        return (layer_is_swa(il) && m_head_size_swa > 0) ? m_head_size_swa : m_head_size;
    }

    // KV head count for layer `il` (some archs, e.g. Deci-style, vary head_count_kv per layer).
    int n_head_kv(int il) const {
        return (!m_n_head_kv_per_layer.empty() && il < static_cast<int>(m_n_head_kv_per_layer.size()))
                   ? static_cast<int>(m_n_head_kv_per_layer[il])
                   : m_n_head_kv;
    }

    // Attention (softmax) scale for layer `il`. An explicit metadata scale wins; otherwise use
    // 1/sqrt(head_size(il)) so SWA and global layers each get the head-size-correct scale.
    float kq_scale(int il) const {
        return m_attention_scale != 0.0f ? m_attention_scale : 1.0f / std::sqrt(static_cast<float>(head_size(il)));
    }

    // RoPE config for layer `il` (gemma4 SWA layers rope with a different freq_base / n_dims).
    RopeConfig layer_rope_config(int il) const {
        return layer_is_swa(il) ? m_rope_config_swa : m_graph->rope_config;
    }

    const std::map<std::string, GGUFMetaData>& m_config;
    std::unordered_map<std::string, ov::Tensor>& m_weights;
    std::unordered_map<std::string, gguf_tensor_type>& m_qtypes;
    std::shared_ptr<GgufGraph> m_graph;

    int m_n_layer = 0, m_n_head = 0, m_n_head_kv = 0, m_head_size = 0, m_n_embd = 0;
    std::vector<uint32_t> m_n_head_kv_per_layer;  // non-empty when head_count_kv varies by layer
    float m_rms_eps = 0.0f;

    // auto-detected per-architecture structure
    bool m_has_qk_norm = false, m_has_qkv_bias = false, m_has_attn_out_bias = false, m_has_rope_freqs = false;
    bool m_has_fused_qkv = false, m_qk_norm_full = false, m_is_moe = false;
    bool m_has_moe_gate_bias = false, m_moe_softmax_weight = false, m_moe_swiglu_oai = false;
    bool m_has_moe_expert_bias = false, m_has_sinks = false, m_has_swa = false;
    bool m_has_attn_post_norm = false, m_has_ffn_post_norm = false, m_is_geglu = false;
    bool m_has_v_norm = false;  // gemma4: V is also RMSNorm'd like K
    // muse-glimmer specifics
    bool m_has_attn_gate = false;        // sigmoid gate multiplied into the attention output
    bool m_scaleless_embd_norm = false;  // weightless RMSNorm on the token embeddings
    bool m_rope_on_swa_only = false;     // global (non-SWA) layers are NoPE
    float m_post_norm_eps = 0.0f;        // eps for post-attn/post-FFN norms (0 -> m_rms_eps)
    // qwen35 (hybrid Gated-DeltaNet + full attention)
    bool m_is_qwen35 = false;
    int m_ssm_conv_kernel = 0;                     // d_conv
    int m_ssm_state_size = 0;                      // head_k_dim / head_v_dim
    int m_ssm_group_count = 0;                     // num_k_heads
    int m_ssm_dt_rank = 0;                         // num_v_heads
    int m_ssm_inner_size = 0;                      // d_inner = num_v_heads * head_v_dim
    int m_full_attn_interval = 0;                  // full attention when (il + 1) % interval == 0
    int m_n_layer_nextn = 0;                       // trailing MTP blocks, not executed
    std::vector<int32_t> m_rope_sections;          // M-RoPE per-axis section widths
    std::vector<int32_t> m_recurrent_layer_flags;  // explicit per-layer recurrent flags
    // True when layer `il` is a linear-attention (recurrent) layer. An explicit per-layer flag
    // array wins over the interval rule, matching llama.cpp's load order.
    bool is_recurrent_layer(int il) const {
        if (!m_is_qwen35) {
            return false;
        }
        if (il < static_cast<int>(m_recurrent_layer_flags.size())) {
            return m_recurrent_layer_flags[il] != 0;
        }
        return m_full_attn_interval > 0 && ((il + 1) % m_full_attn_interval != 0);
    }
    // Norm key suffixes; overridden for archs that use non-standard naming (exaone4, gpt-oss).
    std::string m_attn_norm_key{"attn_norm.weight"};
    std::string m_ffn_norm_key{"ffn_norm.weight"};
    int m_n_expert = 0, m_n_expert_used = 0, m_n_dense_lead = 0, m_n_expert_shared = 0;
    float m_embedding_scale = 1.0f, m_residual_scale = 1.0f, m_logit_scale = 1.0f, m_attention_scale = 0.0f;
    float m_expert_weights_scale = 0.0f;
    float m_attn_soft_cap = 0.0f, m_final_logit_soft_cap = 0.0f;
    float m_rope_freq_base_swa = 0.0f;
    int m_swa_layer_pattern = 2;
    std::vector<int32_t> m_swa_layer_flags;  // gemma4: per-layer SWA flags (1=SWA, 0=global)
    int m_n_embd_per_layer = 0;              // gemma4: per-layer embedding projection dimension
    int m_shared_kv_layers = 0;              // gemma4: N trailing layers that share KV from earlier layers
    int m_head_size_swa = 0;                 // gemma4: head size for SWA layers (differs from global)
    int m_rope_dim_swa = 0;                  // gemma4: rope dims for SWA layers
    RopeConfig m_rope_config_swa{};
    int m_rope_op_case = ROPE_OP_CASE_NEOX;

    std::map<std::string, ov::PartialShape> m_tensor_shapes;
    std::map<std::string, ov::element::Type> m_tensor_types;
    // Names of weights already emitted as GGML_OP_NONE leaves, so a weight referenced by several
    // ops (or tied, e.g. MQA tie-V) is emitted once.
    std::set<std::string> m_emitted_weights;
};

std::shared_ptr<GgufGraph> TransformerBuilder::build() {
    using ov::element::f32;
    using ov::element::i64;
    const int64_t D = -1;  // dynamic token length, used for model-input Parameters only

    // Per-node output shapes are STATIC, like the cgraph decoder (which builds the graph
    // for a concrete token length). We use a representative token length T; the translators
    // emit dynamic reshapes (-1 / 0) where needed, and MakeStateful + the dynamic input
    // Parameters carry the real dynamic-ness. T affects only the per-node shape metadata.
    const int64_t T = 1;

    using ov::element::i32;
    // ---- Model inputs (names match the cgraph decoder's get_graph_input_ov_name) ----
    // gguf uses i32 for token/position/index inputs.
    add_input("inp_tokens", i32, ps({1, 1, 1, D}));
    add_input("inp_pos", i32, ps({1, 1, 1, D}));
    add_input("inp_out_ids", i32, ps({1, 1, 1, D}));
    add_input("self_kq_mask", ov::element::f32, ps({1, 1, D, D}));
    // gpt-oss alternates sliding-window / full attention; the windowed mask is a separate
    // input. Only added when the model uses SWA.
    if (m_has_swa) {
        add_input("self_kq_mask_swa", ov::element::f32, ps({1, 1, D, D}));
    }
    // No beam_idx here. It is a beam-search cache-reorder index for an OpenVINO STATEFUL cache,
    // which ggml has no equivalent of -- so the cgraph decoder does not produce one either, and a
    // builder that declared it would give the two decoders different stateless IO. The pass that
    // creates the state creates it (MakeStateful), because that is where its only consumer, the
    // Gather on the past, is emitted.

    // KV-cache update index (consumed by SET_ROWS; unused in the stateful Concat branch).
    add_input("inp_kv_idx", i32, ps({1, 1, 1, D}));
    m_tensor_shapes["inp_kv_idx"] = ps({1, 1, 1, T});
    m_tensor_types["inp_kv_idx"] = i32;

    // token_len_per_seq: number of new tokens per sequence; used by TranslateSession's mask
    // slicing (add_sliced_mask) to build KQ_mask_sliced. An extra (Parameter) input.
    {
        auto p = std::make_shared<ov::op::v0::Parameter>(i64, ps({1}));
        p->set_friendly_name("token_len_per_seq");
        p->output(0).set_names({"token_len_per_seq"});
        m_graph->model_extra_inputs["token_len_per_seq"] = p;
    }

    // ---- Embedding: GET_ROWS(token_embd.weight, inp_tokens) -> "embd" ----
    add_weight("token_embd.weight");
    m_tensor_shapes["inp_tokens"] = ps({1, 1, 1, T});
    std::string cur =
        add_op("GGML_OP_GET_ROWS", "embd", {"token_embd.weight", "inp_tokens"}, ps({1, 1, T, m_n_embd}), f32);
    // MiniCPM scales the embeddings by a constant.
    if (m_embedding_scale != 1.0f) {
        cur = scale(cur, m_embedding_scale, "embd_scaled");
    }
    // muse-glimmer normalizes the token embeddings with a WEIGHTLESS RMSNorm before layer 0
    // (build_norm with a null weight -> plain ggml_rms_norm, no multiplicative term).
    if (m_scaleless_embd_norm) {
        cur = add_op("GGML_OP_RMS_NORM", "embd_normed", {cur}, m_tensor_shapes.at(cur), f32, 0, {{"eps", m_rms_eps}});
    }

    // rope freq factors (llama-3 long context, phi-3): an optional 3rd ROPE input.
    if (m_has_rope_freqs) {
        add_weight("rope_freqs.weight");
    }

    // kq_scale, head size, KV heads and rope config are all computed per-layer inside the loop
    // via the accessors above (kq_scale(il) etc.), so SWA and global layers each get the correct
    // head-size-dependent values (e.g. gemma4 SWA layers use head_size_swa).

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
    if (m_n_embd_per_layer > 0) {
        add_weight("per_layer_token_embd.weight");
        add_weight("per_layer_model_proj.weight");
        add_weight("per_layer_proj_norm.weight");
        const int pe_total = m_n_embd_per_layer * m_n_layer;

        // Token embedding lookup: [1,1,T, pe_total] -> reshape to [1, n_layer, T, n_embd_per_layer]
        auto pe_flat = add_op("GGML_OP_GET_ROWS",
                              "pe_tok_flat",
                              {"per_layer_token_embd.weight", "inp_tokens"},
                              ps({1, 1, T, pe_total}),
                              f32);
        const float pe_scale = std::sqrt(static_cast<float>(m_n_embd_per_layer));
        pe_flat = scale(pe_flat, pe_scale, "pe_tok_flat_scaled");
        auto pe_tok = reshape_to_layer_major(pe_flat, "pe_tok");

        // Model projection: MUL_MAT(per_layer_model_proj, embd) -> [1,1,T, pe_total]
        // per_layer_model_proj is [n_embd, pe_total] -> output [pe_total] per token
        auto proj_flat = add_op("GGML_OP_MUL_MAT",
                                "pe_proj_flat",
                                {"per_layer_model_proj.weight", cur},
                                ps({1, 1, T, pe_total}),
                                f32);
        const float proj_scale = 1.0f / std::sqrt(static_cast<float>(m_n_embd));
        proj_flat = scale(proj_flat, proj_scale, "pe_proj_flat_scaled");

        // Reshape to [1, n_layer, T, n_embd_per_layer] for per-slice RMS_NORM
        auto proj_4d = reshape_to_layer_major(proj_flat, "pe_proj_4d");

        // RMS_NORM + per_layer_proj_norm weight (applied over last dim = n_embd_per_layer)
        auto proj_norm = add_op("GGML_OP_RMS_NORM",
                                "pe_proj_rms",
                                {proj_4d},
                                ps({1, m_n_layer, T, m_n_embd_per_layer}),
                                f32,
                                0,
                                {{"eps", m_rms_eps}});
        m_tensor_shapes["per_layer_proj_norm.weight"] = ps({1, 1, 1, m_n_embd_per_layer});
        auto proj_normed = add_op("GGML_OP_MUL",
                                  "pe_proj_normed",
                                  {proj_norm, "per_layer_proj_norm.weight"},
                                  ps({1, m_n_layer, T, m_n_embd_per_layer}),
                                  f32);

        // Sum token embd + projection, scale by 1/sqrt(2)
        auto pe_sum =
            add_op("GGML_OP_ADD", "pe_sum", {proj_normed, pe_tok}, ps({1, m_n_layer, T, m_n_embd_per_layer}), f32);
        const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
        add_op("GGML_OP_SCALE",
               "per_layer_embd",
               {pe_sum},
               ps({1, m_n_layer, T, m_n_embd_per_layer}),
               f32,
               0,
               {{"scale", inv_sqrt2}, {"bias", 0.0f}});
    }

    // Gemma4 shared-KV: SWA shared layers reuse a different anchor than global shared layers.
    // Precompute the last own-KV layer for each SWA type (mirrors llama's layer_reuse_cb).
    const int n_own_kv = m_n_layer - m_shared_kv_layers;  // first n_own_kv layers have own KV
    int anchor_swa = -1, anchor_global = -1;
    if (m_shared_kv_layers > 0) {
        for (int i = n_own_kv - 1; i >= 0; --i) {
            bool i_is_swa = layer_is_swa(i);
            if (anchor_swa < 0 && i_is_swa)
                anchor_swa = i;
            if (anchor_global < 0 && !i_is_swa)
                anchor_global = i;
            if (anchor_swa >= 0 && anchor_global >= 0)
                break;
        }
    }

    for (int il = 0; il < m_n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const std::string inpSA = cur;

        // attn_norm (key varies by arch: "attn_norm.weight" for most, "post_attention_norm.weight" for exaone4)
        std::string attn_norm = rms_norm(cur, p + m_attn_norm_key, p + "attn_norm");

        // Per-layer hyperparameters (SWA vs global head size / rope / scale; per-layer KV heads).
        // See the accessor definitions above for the arch-specific derivation rules.
        const bool is_swa_layer = layer_is_swa(il);
        const int head_size_l = head_size(il);
        const int n_head_kv_l = n_head_kv(il);
        const float kq_scale = this->kq_scale(il);
        const RopeConfig rope_config_l = layer_rope_config(il);

        // Q/K/V projections: MUL_MAT(w, attn_norm), then conceptual reshape to heads.
        // Fused-QKV archs (phi-3, minicpm) carry a single attn_qkv weight; split it into
        // separate q/k/v weights so the rest of the layer is architecture-agnostic.
        if (m_is_qwen35 && is_recurrent_layer(il)) {
            // Hybrid stack: 3 of every 4 layers replace attention with a Gated DeltaNet block.
            // It produces the same thing the attention path does -- the sublayer output before
            // the residual -- so the shared FFN tail below is reused verbatim.
            const std::string gdn_out = build_qwen35_gdn(il, attn_norm, T);
            std::string sa = inpSA;
            std::string ao = gdn_out;
            if (il == m_n_layer - 1) {
                ao = add_op("GGML_OP_GET_ROWS",
                            p + "attn_out_g",
                            {gdn_out, "inp_out_ids"},
                            ps({1, 1, T, m_n_embd}),
                            f32);
                sa = add_op("GGML_OP_GET_ROWS", p + "inpSA_g", {inpSA, "inp_out_ids"}, ps({1, 1, T, m_n_embd}), f32);
            }
            auto ffn_inp_r = add_op("GGML_OP_ADD", p + "ffn_inp", {ao, sa}, ps({1, 1, T, m_n_embd}), f32);
            auto ffn_norm_r = rms_norm(ffn_inp_r, p + m_ffn_norm_key, p + "ffn_norm");
            auto down_r = build_dense_ffn(p, ffn_norm_r, T);
            cur = add_op("GGML_OP_ADD", p + "l_out", {down_r, ffn_inp_r}, ps({1, 1, T, m_n_embd}), f32);
            continue;
        }
        if (m_is_qwen35) {
            register_qwen35_q_gate(il);
            add_weight(p + "attn_k.weight");
            add_weight(p + "attn_v.weight");
        } else if (m_has_fused_qkv) {
            register_fused_qkv(il);
        } else {
            add_weight(p + "attn_q.weight");
            add_weight(p + "attn_k.weight");
            // MQA tie-V: some global attention layers (e.g. gemma4-12B every 6th layer) have
            // head_count_kv==1 and no separate attn_v.weight; V shares K's weight tensor.
            if (m_weights.count(p + "attn_v.weight") > 0) {
                add_weight(p + "attn_v.weight");
            } else {
                // MQA tie-V: emit a separate attn_v.weight GGML_OP_NONE leaf that reuses K's
                // extracted tensors, so attn_v.weight resolves in the tensor map like any weight.
                add_weight_from(p + "attn_v.weight", p + "attn_k");
            }
        }
        auto q = add_op("GGML_OP_MUL_MAT",
                        p + "Qcur",
                        {p + "attn_q.weight", attn_norm},
                        ps({1, 1, T, head_size_l * m_n_head}),
                        f32);
        auto k = add_op("GGML_OP_MUL_MAT",
                        p + "Kcur",
                        {p + "attn_k.weight", attn_norm},
                        ps({1, 1, T, head_size_l * n_head_kv_l}),
                        f32);
        auto v = add_op("GGML_OP_MUL_MAT",
                        p + "Vcur",
                        {p + "attn_v.weight", attn_norm},
                        ps({1, 1, T, head_size_l * n_head_kv_l}),
                        f32);

        // Q/K/V projection biases (qwen2 / qwen2.5: separate attn_{q,k,v}.bias).
        if (m_has_qkv_bias && !m_has_fused_qkv) {
            q = add_bias(q, p + "attn_q.bias", p + "Qcur_b");
            k = add_bias(k, p + "attn_k.bias", p + "Kcur_b");
            v = add_bias(v, p + "attn_v.bias", p + "Vcur_b");
        }

        // Full-width q/k norm (OLMoE): normalize the whole projection before splitting heads.
        if (m_has_qk_norm && m_qk_norm_full) {
            q = rms_norm(q, p + "attn_q_norm.weight", p + "Qcur_normed");
            k = rms_norm(k, p + "attn_k_norm.weight", p + "Kcur_normed");
        }

        // reshape Q/K/V to [1, n_tokens, n_head(_kv), head_size]
        q = add_op("GGML_OP_RESHAPE", p + "Qcur_r", {q}, ps({1, T, m_n_head, head_size_l}), f32, 1);
        k = add_op("GGML_OP_RESHAPE", p + "Kcur_r", {k}, ps({1, T, n_head_kv_l, head_size_l}), f32, 1);
        v = add_op("GGML_OP_RESHAPE", p + "Vcur_r", {v}, ps({1, T, n_head_kv_l, head_size_l}), f32, 1);

        // per-head q_norm / k_norm (qwen3, hunyuan, gemma4)
        if (m_has_qk_norm && !m_qk_norm_full) {
            q = rms_norm(q, p + "attn_q_norm.weight", p + "Qcur_normed");
            k = rms_norm(k, p + "attn_k_norm.weight", p + "Kcur_normed");
        }
        // gemma4: V gets a plain RMSNorm (no multiplicative weight, just normalize).
        if (m_has_v_norm) {
            v = add_op("GGML_OP_RMS_NORM",
                       p + "Vcur_normed",
                       {v},
                       ps({1, T, n_head_kv_l, head_size_l}),
                       f32,
                       0,
                       {{"eps", m_rms_eps}});
        }

        // RoPE (NEOX). rope_freqs.weight (per-dim frequency factor) is an optional 3rd input.
        // For gemma4: global layers use rope_freqs (proportional/NTK scaling), SWA layers don't.
        // muse-glimmer ropes ONLY its sliding-window layers; its global layers are NoPE
        // (llama.cpp muse-glimmer.cpp: `const bool use_rope = hparams.is_swa(il)`).
        const bool use_rope = !m_rope_on_swa_only || is_swa_layer;
        const bool use_rope_freqs = m_has_rope_freqs && !is_swa_layer;
        const std::vector<std::string> q_rope_in = use_rope_freqs
                                                       ? std::vector<std::string>{q, "inp_pos", "rope_freqs.weight"}
                                                       : std::vector<std::string>{q, "inp_pos"};
        const std::vector<std::string> k_rope_in = use_rope_freqs
                                                       ? std::vector<std::string>{k, "inp_pos", "rope_freqs.weight"}
                                                       : std::vector<std::string>{k, "inp_pos"};
        if (use_rope) {
            q = add_op("GGML_OP_ROPE",
                       p + "Qcur_rope",
                       q_rope_in,
                       ps({1, T, m_n_head, head_size_l}),
                       f32,
                       m_rope_op_case,
                       {{"rope_config", rope_config_l}});
            k = add_op("GGML_OP_ROPE",
                       p + "Kcur_rope",
                       k_rope_in,
                       ps({1, T, n_head_kv_l, head_size_l}),
                       f32,
                       m_rope_op_case,
                       {{"rope_config", rope_config_l}});
        }

        // ---- KV cache store ----
        // Gemma4: layers with shared_kv_layers have no K/V of their own; they reuse the KV
        // from the last layer of the same SWA type that has its own KV cache. SWA layers
        // reuse the last own-KV SWA layer; global layers reuse the last own-KV global layer.
        const bool has_own_kv = (m_shared_kv_layers == 0) || (il < n_own_kv);
        int anchor_il = il;
        if (!has_own_kv) {
            anchor_il = is_swa_layer ? anchor_swa : anchor_global;
            if (anchor_il < 0)
                anchor_il = n_own_kv - 1;  // fallback
        }
        const std::string kc = "cache_k_l" + std::to_string(anchor_il);
        const std::string vc = "cache_v_l" + std::to_string(anchor_il);

        if (has_own_kv) {
            // Per-layer f16 KV cache Parameters. The K/V read back out of a cache is f16, and so is
            // Q after its convert in the translator, so the FLASH_ATTN inputs agree.
            const ov::PartialShape cache_shape = ps({1, D, n_head_kv_l, head_size_l});
            if (!m_graph->model_inputs.count(kc)) {
                add_input(kc, ov::element::f16, cache_shape);
                add_input(vc, ov::element::f16, cache_shape);
            }
            m_tensor_shapes[kc] = ps({1, T, n_head_kv_l, head_size_l});
            m_tensor_shapes[vc] = ps({1, T, n_head_kv_l, head_size_l});
            m_tensor_types[kc] = ov::element::f16;
            m_tensor_types[vc] = ov::element::f16;

            // SET_ROWS(cur, idx, cache) -> the cache with this step's rows written in. Lowered by
            // the frontend to a stateless ScatterUpdate, or by the caller-registered MakeStateful
            // extension to a ReadValue/Concat/Assign OpenVINO state.
            k = add_op("GGML_OP_SET_ROWS",
                       kc,
                       {k, "inp_kv_idx", kc},
                       ps({1, T, n_head_kv_l, head_size_l}),
                       ov::element::f16);
            v = add_op("GGML_OP_SET_ROWS",
                       vc,
                       {v, "inp_kv_idx", vc},
                       ps({1, T, n_head_kv_l, head_size_l}),
                       ov::element::f16);
            // The written-through caches are model outputs, so the stateless graph returns each
            // updated cache as a Result (which MakeStateful, when registered, turns into an Assign
            // sink paired with the cache's ReadValue).
            m_graph->model_output_names.push_back(kc);
            m_graph->model_output_names.push_back(vc);
        } else {
            // Shared-KV layer: K/V have already been set in the anchor layer's SET_ROWS.
            // Use the anchor's combined cache. If the current layer has a smaller head size
            // (SWA shared layer vs a global anchor), slice K/V to the layer's head_size along
            // the last dim, mirroring llama.cpp's ggml_view_4d with n_embd_head_k(il).
            const ov::PartialShape& anchor_kc_shape = m_tensor_shapes.at(kc);  // [1, T, n_kv, anchor_head]
            const int64_t anchor_hs = anchor_kc_shape[3].is_static() ? anchor_kc_shape[3].get_length() : m_head_size;
            if (head_size_l < static_cast<int>(anchor_hs)) {
                // Shrink the last (head-size) axis to head_size_l. This is a plain single-axis shrink
                // at offset 0, which is exactly what the shared VIEW op_case 3 does -- and what the
                // cgraph decoder assigns to llama.cpp's corresponding ggml_view_4d with
                // n_embd_head_k(il) -- so describe it the way that case expects rather than with a
                // builder-only case: "view_slice" = {ov_axis, start, len} plus the input's own shape.
                // No "view_reshape": the sliced shape already is this node's output shape.
                const ov::PartialShape slice_shape = ps({1, T, n_head_kv_l, head_size_l});
                const std::vector<int64_t> hs_slice{3, 0, int64_t(head_size_l)};
                k = add_op("GGML_OP_VIEW",
                           p + "k_hslice",
                           {kc},
                           slice_shape,
                           ov::element::f16,
                           3,
                           {{"view_slice", hs_slice}, {"input_ggml_shape", shape_of(kc)}});
                v = add_op("GGML_OP_VIEW",
                           p + "v_hslice",
                           {vc},
                           slice_shape,
                           ov::element::f16,
                           3,
                           {{"view_slice", hs_slice}, {"input_ggml_shape", shape_of(vc)}});
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
        if (m_has_sinks) {
            add_named_weight(p + "attn_sinks.weight");
            attn_in.push_back(p + "attn_sinks.weight");
        }
        std::map<std::string, ov::Any> attn_attrs = {{"scale", kq_scale}};
        if (m_attn_soft_cap != 0.0f) {
            attn_attrs["kq_soft_cap"] = m_attn_soft_cap;
        }
        auto attn = add_op("GGML_OP_FLASH_ATTN_EXT",
                           p + "kqv",
                           attn_in,
                           ps({1, T, m_n_head, head_size_l}),
                           f32,
                           100,  // builder layout: q/k/v are ggml-natural [1, T, n_head, head_size]
                           std::move(attn_attrs));

        // reshape back to [1, 1, n_tokens, n_head*head_size]
        auto attn_2d =
            add_op("GGML_OP_RESHAPE", p + "kqv_merged", {attn}, ps({1, 1, T, m_n_head * head_size_l}), f32, 2);

        // muse-glimmer: sigmoid output gate. The gate is a projection of the PRE-attention
        // normed hidden (the same `attn_norm` tensor Q/K/V come from), squashed by sigmoid and
        // multiplied elementwise into the merged attention output before the wo projection.
        if (m_has_attn_gate) {
            add_weight(p + "attn_gate.weight");
            auto gate = add_op("GGML_OP_MUL_MAT",
                               p + "attn_gate",
                               {p + "attn_gate.weight", attn_norm},
                               ps({1, 1, T, m_n_head * head_size_l}),
                               f32);
            gate = add_op("GGML_UNARY_OP_SIGMOID", p + "attn_gate_sig", {gate}, m_tensor_shapes.at(gate), f32);
            attn_2d = add_op("GGML_OP_MUL", p + "kqv_gated", {attn_2d, gate}, m_tensor_shapes.at(attn_2d), f32);
        }

        // output projection (+ optional bias)
        add_weight(p + "attn_output.weight");
        auto attn_out = add_op("GGML_OP_MUL_MAT",
                               p + "attn_out",
                               {p + "attn_output.weight", attn_2d},
                               ps({1, 1, T, m_n_embd}),
                               f32);
        if (m_has_attn_out_bias) {
            attn_out = add_bias(attn_out, p + "attn_output.bias", p + "attn_out_b");
        }
        // MiniCPM scales the attention sublayer output before the residual add.
        if (m_residual_scale != 1.0f) {
            attn_out = scale(attn_out, m_residual_scale, p + "attn_out_scaled");
        }
        std::string sa = inpSA;
        std::string ao = attn_out;
        if (il == m_n_layer - 1) {
            ao = add_op("GGML_OP_GET_ROWS", p + "attn_out_g", {attn_out, "inp_out_ids"}, ps({1, 1, T, m_n_embd}), f32);
            sa = add_op("GGML_OP_GET_ROWS", p + "inpSA_g", {inpSA, "inp_out_ids"}, ps({1, 1, T, m_n_embd}), f32);
        }
        // Gemma2: post-attention RMSNorm applied to the sublayer output before residual add.
        // Applied after GET_ROWS so the selected-token path matches gemma2.cpp's order.
        if (m_has_attn_post_norm) {
            ao = rms_norm(ao, p + "post_attention_norm.weight", p + "attn_post_norm", m_post_norm_eps);
        }

        auto ffn_inp = add_op("GGML_OP_ADD", p + "ffn_inp", {ao, sa}, ps({1, 1, T, m_n_embd}), f32);

        // Pre-FFN/MoE norm. Key varies by arch (m_ffn_norm_key is set in constructor).
        auto ffn_norm = rms_norm(ffn_inp, p + m_ffn_norm_key, p + "ffn_norm");

        // Hybrid MoE: lead layers (il < m_n_dense_lead) are always dense regardless of m_is_moe.
        const bool is_moe_layer = m_is_moe && (il >= m_n_dense_lead);
        std::string down = is_moe_layer ? build_moe_ffn(p, ffn_norm, T)
                           : m_is_geglu ? build_geglu_ffn(p, ffn_norm, T)
                                        : build_dense_ffn(p, ffn_norm, T);
        // MiniCPM scales the FFN sublayer output before the residual add.
        if (m_residual_scale != 1.0f) {
            down = scale(down, m_residual_scale, p + "ffn_out_scaled");
        }
        // Gemma2: post-FFN RMSNorm applied to the FFN output before residual add.
        if (m_has_ffn_post_norm) {
            down = rms_norm(down, p + "post_ffw_norm.weight", p + "ffn_post_norm", m_post_norm_eps);
        }

        cur = add_op("GGML_OP_ADD", p + "l_out", {down, ffn_inp}, ps({1, 1, T, m_n_embd}), f32);

        // Gemma4: per-layer embedding injection.
        // Each layer takes a slice of the pre-projected per-layer embedding (shape
        // [1,1,T,n_embd_per_layer]), gates it through inp_gate.weight (GELU), multiplies
        // by the per-layer slice, projects back to n_embd, post-norms, then adds residual.
        // per_layer_token_embd (global) is pre-projected before the loop in build().
        if (m_n_embd_per_layer > 0) {
            // Global per-layer projected embedding was added as "per_layer_embd" before the loop.
            // Slice out this layer's [1,1,T,n_embd_per_layer] chunk (dim 1 at index il).
            // The second input is passed purely as a shape reference: per_layer_embd is stored
            // layer-major, so the slice's token axis does not necessarily sit where the layer
            // activation keeps its own, and the ops below combine the two elementwise. See the
            // op_case 104 note in op/view.cpp.
            //
            // It has to be a tensor that still carries every token. At the last layer `cur` has
            // already been filtered down to the output rows by the GET_ROWS above, while
            // per_layer_embd always holds all T of them, so using `cur` there would ask op_case
            // 104 to reinterpret T*D values into 1*D. Use the layer's unfiltered input instead
            // and let the GET_ROWS below do the filtering.
            const std::string& pl_shape_ref = (il == m_n_layer - 1) ? inpSA : cur;
            const std::string pl_slice = p + "per_layer_slice";
            add_op("GGML_OP_VIEW",
                   pl_slice,
                   {"per_layer_embd", pl_shape_ref},
                   ps({1, 1, T, m_n_embd_per_layer}),
                   f32,
                   104,  // layer-index slice using the "layer_idx" attribute
                   {{"layer_idx", int64_t(il)}});

            // At the last layer, cur is already filtered to 1 token via inp_out_ids.
            // Mirror llama.cpp gemma4.cpp:347-349: also filter pl_slice so the MUL
            // doesn't broadcast it back to the full sequence length.
            std::string pl_slice_used = pl_slice;
            if (il == m_n_layer - 1) {
                pl_slice_used = p + "per_layer_slice_sel";
                add_op("GGML_OP_GET_ROWS",
                       pl_slice_used,
                       {pl_slice, "inp_out_ids"},
                       ps({1, 1, T, m_n_embd_per_layer}),
                       f32);
            }

            // gate: cur -> inp_gate.weight -> GELU -> [1,1,T,n_embd_per_layer]
            add_weight(p + "inp_gate.weight");
            auto gated = add_op("GGML_OP_MUL_MAT",
                                p + "inp_gate_mm",
                                {p + "inp_gate.weight", cur},
                                ps({1, 1, T, m_n_embd_per_layer}),
                                f32);
            gated = add_op("GGML_UNARY_OP_GELU", p + "inp_gate_gelu", {gated}, ps({1, 1, T, m_n_embd_per_layer}), f32);

            // elementwise multiply by per-layer slice
            auto mul_pe =
                add_op("GGML_OP_MUL", p + "pe_mul", {gated, pl_slice_used}, ps({1, 1, T, m_n_embd_per_layer}), f32);

            // project back to n_embd
            add_weight(p + "proj.weight");
            auto pe_proj =
                add_op("GGML_OP_MUL_MAT", p + "pe_proj", {p + "proj.weight", mul_pe}, ps({1, 1, T, m_n_embd}), f32);

            // post-norm + residual add
            pe_proj = rms_norm(pe_proj, p + "post_norm.weight", p + "pe_post_norm");
            cur = add_op("GGML_OP_ADD", p + "pe_out", {cur, pe_proj}, ps({1, 1, T, m_n_embd}), f32);
        }

        // Gemma4: per-layer output scale (layer_output_scale.weight is a scalar [1]).
        if (m_weights.count(p + "layer_output_scale.weight")) {
            add_named_weight(p + "layer_output_scale.weight");
            cur = add_op("GGML_OP_MUL",
                         p + "scaled_out",
                         {cur, p + "layer_output_scale.weight"},
                         m_tensor_shapes.at(cur),
                         f32);
        }
    }

    // final norm + lm_head
    cur = rms_norm(cur, "output_norm.weight", "result_norm");

    const std::string lm_head_w = m_weights.count("output.weight") ? "output.weight" : "token_embd.weight";
    add_weight(lm_head_w);
    const int n_vocab = static_cast<int>(weight_tensor(lm_head_w).get_shape()[0]);  // rows = vocab
    auto logits = add_op("GGML_OP_MUL_MAT", "result_output", {lm_head_w, cur}, ps({1, 1, T, n_vocab}), f32);
    // MiniCPM scales the logits (1/(n_embd/dim_model_base)).
    if (m_logit_scale != 1.0f) {
        logits = scale(logits, m_logit_scale, "result_output_scaled");
    }
    // Gemma2/Gemma3 final logit soft-cap: tanh(logits / cap) * cap.
    if (m_final_logit_soft_cap != 0.0f) {
        logits = scale(logits, 1.0f / m_final_logit_soft_cap, "logits_softcap_scaled");
        logits = add_op("GGML_UNARY_OP_TANH", "logits_softcap_tanh", {logits}, m_tensor_shapes.at(logits), f32);
        logits = scale(logits, m_final_logit_soft_cap, "result_output_softcapped");
    }

    // logits is the primary output; the cache_k/v_l* outputs (appended per layer above)
    // become Assign sinks via MakeStateful.
    m_graph->model_output_names.insert(m_graph->model_output_names.begin(), logits);
    return m_graph;
}

}  // namespace

// Architectures END-TO-END VERIFIED on the generic builder: convert + compile + generation
// checked against the reference (native llama.cpp / HF) on a real checkpoint. Safe to rely on.
const std::set<std::string>& verified_archs() {
    static const std::set<std::string> archs = {
        "llama",  // llama-2 / llama-3
        "qwen2",  // qwen2 / qwen2.5
        "qwen3",
        "phi3",     // phi-3 (fused QKV)
        "minicpm",  // NORMAL rope + scalar scales
        "hunyuan-dense",
        "olmoe",     // OLMoE 1B-7B (MoE)
        "qwen3moe",  // Qwen3 MoE: same topology as olmoe
        // Qwen3.5/3.6 hybrid: Gated-DeltaNet linear attention on 3 of every 4 layers, full
        // attention with M-RoPE and an interleaved query+gate projection on the rest. Verified
        // token-exact against llama.cpp on Qwen3.5-0.8B-Q8_0 and Ternary-Bonsai-27B-Q2_g64.
        // GREEDY / BATCH 1 ONLY: the recurrent conv and delta states are not reordered by
        // beam_idx and have a static batch of 1, so beam search or batch > 1 fails at inference
        // (a Concat shape mismatch on the conv window) rather than producing wrong output.
        "qwen35",
        "gpt-oss",  // MoE + sinks + SWA
        "gemma",    // Gemma 2B / 7B
        "gemma2",   // Gemma 2: post-norms + attention soft-cap
        "gemma3",   // Gemma 3: post-norms + final logit soft-cap
        "gemma4",   // Gemma 4: SWA, per-layer embeddings, shared KV
    };
    return archs;
}

// Architectures EXPECTED to work via the generic builder's GGUF-tensor-table auto-detection, but
// NOT yet end-to-end verified on a real checkpoint. Enabled (they convert), but conversion emits a
// one-time warning so callers know they are best-effort. Promote to verified_archs() once a model
// of the family has been checked against the reference. See docs/adding_an_architecture.md.
const std::set<std::string>& experimental_archs() {
    static const std::set<std::string> archs = {
        // H2 2025: dense
        "llama-embed",  // Bidirectional LLaMA (embedding, no causal mask)
        "exaone4",      // EXAONE 4.0: NEOX rope, post-norms (attn+ffn)
        "plamo3",       // PLaMo-3: NEOX rope, post-norms (attn+ffn)
        "smollm3",      // SmolLM3: NORMAL rope + SWA
        // H2 2025: MoE
        "hunyuan-moe",   // Hunyuan MoE: NEOX rope, MoE routing, QK-norm
        "glm4moe",       // GLM 4.5 MoE: NEOX rope, 1 dense lead layer, MoE + attn post-norm
        "exaone-moe",    // EXAONE MoE: NEOX rope, SWA + MoE, shared expert
        "minimax-m2",    // Minimax M2: NEOX rope, pure MoE
        "ernie4_5-moe",  // Ernie 4.5 MoE: NORMAL rope, dense lead layers + MoE stride
        "bailingmoe2",   // BailingMoe V2: NEOX rope, MoE + shared expert + QK-norm
        // 2026: dense
        "maincoder",     // Maincoder-1B: NORMAL rope, QK-norm (auto-detected)
        "mistral3",      // Ministral-3B: NORMAL rope, dense
        "muse-glimmer",  // Muse Glimmer (Meta Onyx): NORMAL rope on SWA layers only (global
                         // layers are NoPE), sigmoid attention output gate, QK-norm,
                         // pre+post norms, final logit soft-cap
        // 2026: MoE
        "mellum",         // JetBrains Mellum: NEOX rope, pure MoE
        "deepseek2-ocr",  // DeepSeekOCR: NEOX rope, dense lead layers + MoE
        "jais2",          // JAIS-2: NEOX rope, dense (biases auto-detected)
    };
    return archs;
}

// All architectures the native builder accepts = verified + experimental.
const std::set<std::string>& supported_archs() {
    static const std::set<std::string> archs = [] {
        std::set<std::string> a = verified_archs();
        a.insert(experimental_archs().begin(), experimental_archs().end());
        return a;
    }();
    return archs;
}

// Pull the `tokenizer.*` ggml metadata into an ov::AnyMap keyed by the sub-key after the last
// dot (e.g. "tokenizer.ggml.tokens" -> "tokens", "tokenizer.chat_template" -> "chat_template").
// Each GGUF metadata variant is mapped to the ov::Any types a downstream tokenizer builder
// consumes: std::string / std::vector<std::string> / ov::Tensor (arrays and shape-{} scalars).
static ov::AnyMap extract_tokenizer_config(const std::unordered_map<std::string, GGUFMetaData>& metadata) {
    const std::string prefix = "tokenizer.";
    ov::AnyMap cfg;
    for (const auto& [key, value] : metadata) {
        if (key.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        const auto sub_key = key.substr(key.find_last_of('.') + 1);
        std::visit(
            [&](const auto& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::monostate>) {
                    // skip empty
                } else {
                    cfg[sub_key] = v;
                }
            },
            value);
    }
    return cfg;
}

std::shared_ptr<GgufGraph> build_ggml_graph_from_gguf(const std::string& file) {
    auto [metadata, weights, qtypes, mmap, quant_buf] = get_gguf_data(file);
    auto config = config_from_meta(metadata);

    const std::string arch = std::get<std::string>(config.at("architecture"));
    OPENVINO_ASSERT(supported_archs().count(arch),
                    "[GGUF] native GGUF builder does not support architecture '",
                    arch,
                    "'. See supported_archs() in gguf_builder.cpp for the full list.");
    if (experimental_archs().count(arch)) {
        OPENVINO_WARN("[GGUF] architecture '",
                      arch,
                      "' is experimental: it is built by structural auto-detection but has not been "
                      "end-to-end verified against a reference. Validate accuracy before relying on it.");
    }

    TransformerBuilder builder(config, weights, qtypes);
    auto graph = builder.build();
    graph->tokenizer_config = extract_tokenizer_config(metadata);
    return graph;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
