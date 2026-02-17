// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/opsets/opset11.hpp"

namespace ov {
namespace test {
namespace npuw {

// ============================================================================
// Result types (namespace-level)
// ============================================================================

struct KVCacheResult {
    ov::Output<ov::Node> concatenated;
    ov::Output<ov::Node> beam_gather;  // Past KV after beam reorder (before concat)
    std::shared_ptr<ov::Node> assign;  // Sink node for stateful model
};

/// Read-side state from a KV cache Variable (before concat/assign)
struct KVCacheReadState {
    std::shared_ptr<ov::op::util::Variable> variable;
    ov::Output<ov::Node> beam_gather;  // ReadValue -> Gather(beam_idx)
};

/// Result from building a decoder layer or attention block
struct LayerResult {
    ov::Output<ov::Node> output;
    std::vector<std::shared_ptr<ov::Node>> sinks;  // Assign nodes for stateful KV cache
};

// ============================================================================
// Functor type aliases
// ============================================================================

/// Weight functor: construction captures format-specific params;
/// call takes (name, shape, compute_precision) -> output
using WeightFn = std::function<ov::Output<ov::Node>(const std::string&, const ov::Shape&, ov::element::Type)>;

/// Norm functor: construction captures hidden_size, precision, eps;
/// call takes (input, name) -> output
using NormFn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const std::string&)>;

/// FFN functor: construction captures hidden_size, intermediate_size, precision, weight_fn;
/// call takes (input, name) -> output
using FFNFn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const std::string&)>;

/// RoPE functor: construction captures head_dim, precision, and position_ids;
/// call takes (input, name) -> output.  Position IDs are baked in at construction.
using RoPEFn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const std::string&)>;

// ============================================================================
// Weight functors
// ============================================================================

/// Float weights (f32 or f16 storage) with Convert to compute precision when types differ
struct FloatWeight {
    ov::element::Type storage_type;

    FloatWeight(ov::element::Type st = ov::element::f32) : storage_type(st) {}

    ov::Output<ov::Node> operator()(const std::string& name,
                                    const ov::Shape& shape,
                                    ov::element::Type compute_precision) const;
};

/// Convenience aliases
using FP32Weight = FloatWeight;
struct FP16Weight : FloatWeight {
    FP16Weight() : FloatWeight(ov::element::f16) {}
};

/// Compressed integer weights -> Convert(f16) -> Multiply(f16 scale)
/// Matches DCOFF SymmNoZP pattern. Parameterized by storage element type (i8, i4, u4, etc.)
/// When group_size > 0, builds a group quantization pattern:
///   Constant(storage) -> Convert(f16) -> Reshape[rows, cols/gs, gs] -> Multiply(scale[rows, cols/gs, 1]) -> Reshape[rows, cols]
struct CompressedWeight {
    ov::element::Type storage_type;
    size_t group_size;  ///< 0 = per-channel scale, >0 = per-group scale

    explicit CompressedWeight(ov::element::Type st, size_t gs = 0) : storage_type(st), group_size(gs) {}

    ov::Output<ov::Node> operator()(const std::string& name,
                                    const ov::Shape& shape,
                                    ov::element::Type compute_precision) const;
};

/// i8 compressed weights (per-channel)
struct INT8Weight : CompressedWeight {
    INT8Weight() : CompressedWeight(ov::element::i8) {}
};

/// i4 compressed weights (per-channel)
struct INT4Weight : CompressedWeight {
    INT4Weight() : CompressedWeight(ov::element::i4) {}
};

/// i4 group-quantized weights
struct INT4GroupWeight : CompressedWeight {
    explicit INT4GroupWeight(size_t gs = 128) : CompressedWeight(ov::element::i4, gs) {}
};

// ============================================================================
// Norm functors
// ============================================================================

/// Layer normalization (MVN + scale + bias)
struct LayerNorm {
    size_t hidden_size;
    ov::element::Type precision;
    float eps;

    LayerNorm(size_t hs, ov::element::Type prec = ov::element::f32, float e = 1e-5f)
        : hidden_size(hs),
          precision(prec),
          eps(e) {}

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

/// RMS normalization (scale only, no mean subtraction)
struct RMSNorm {
    size_t hidden_size;
    ov::element::Type precision;
    float eps;

    RMSNorm(size_t hs, ov::element::Type prec = ov::element::f32, float e = 1e-5f)
        : hidden_size(hs),
          precision(prec),
          eps(e) {}

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

// ============================================================================
// RoPE functors
// ============================================================================

/// Half-rotation RoPE: split into first/second half, negate+swap, apply cos/sin.
/// Uses frequency-based Sin/Cos nodes. Position IDs are baked in at construction;
/// cos/sin are computed once and shared across all layers (like real models).
/// operator() only applies the rotation.
struct HalfRotationRoPE {
    size_t head_dim;
    ov::Output<ov::Node> cos_freq, sin_freq;  // pre-built from constructor

    HalfRotationRoPE(size_t head_dim,
                     ov::element::Type precision,
                     const ov::Output<ov::Node>& position_ids);

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input,
                                    const std::string& name) const;
};

/// Interleaved RoPE: interleave odd/even elements.
/// Uses frequency-based Sin/Cos nodes. Position IDs are baked in at construction;
/// cos/sin are computed once and shared across all layers (like real models).
/// operator() only applies the rotation.
struct InterleavedRoPE {
    size_t head_dim;
    ov::Output<ov::Node> cos_freq, sin_freq;  // pre-built from constructor

    InterleavedRoPE(size_t head_dim,
                    ov::element::Type precision,
                    const ov::Output<ov::Node>& position_ids);

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input,
                                    const std::string& name) const;
};

// ============================================================================
// Position IDs helpers
// ============================================================================

/// Create 2D position_ids Parameter [batch, seq].
/// The returned output IS the Parameter node (auto-tracked by build_llm's graph traversal).
ov::Output<ov::Node> make_position_ids_2d();

/// Create 3D position_ids Parameter [3, batch, seq] for m-rope (Qwen2.5-VL).
/// Returns a 2D [batch, seq] slice (section 0) for downstream RoPE consumption.
/// The 3D Parameter is auto-tracked by build_llm's graph traversal.
ov::Output<ov::Node> make_position_ids_3d();

// ============================================================================
// FFN functors
// ============================================================================

/// SwiGLU FFN: gate_proj * silu(up_proj), then down_proj
struct SwiGLU {
    size_t hidden_size;
    size_t intermediate_size;
    ov::element::Type precision;
    WeightFn weight_fn;

    SwiGLU(size_t hs, size_t is, ov::element::Type prec, WeightFn wf)
        : hidden_size(hs),
          intermediate_size(is),
          precision(prec),
          weight_fn(std::move(wf)) {}

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

/// GELU FFN: up_proj -> GELU -> down_proj
struct GELUFn {
    size_t hidden_size;
    size_t intermediate_size;
    ov::element::Type precision;
    WeightFn weight_fn;
    WeightFn bias_fn;

    GELUFn(size_t hs, size_t is, ov::element::Type prec, WeightFn wf, WeightFn bf = {})
        : hidden_size(hs),
          intermediate_size(is),
          precision(prec),
          weight_fn(std::move(wf)),
          bias_fn(std::move(bf)) {}

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

// ============================================================================
// Free building-block functions
// ============================================================================

/// Linear projection (MatMul with weight + optional bias).
/// When bias_fn is provided, bias is created through it (matching weight compression pattern).
ov::Output<ov::Node> make_linear(const ov::Output<ov::Node>& input,
                                 size_t in_features,
                                 size_t out_features,
                                 const std::string& name,
                                 ov::element::Type precision = ov::element::f32,
                                 const WeightFn& weight_fn = FP32Weight{},
                                 const WeightFn& bias_fn = {});

/// Reshape for multi-head: [batch, seq, hidden] -> [batch, seq, heads, head_dim]
ov::Output<ov::Node> make_multihead_reshape(const ov::Output<ov::Node>& input,
                                            size_t num_heads,
                                            size_t head_dim,
                                            const std::string& name);

/// Transpose for attention: [batch, seq, heads, dim] <-> [batch, heads, seq, dim]
ov::Output<ov::Node> make_attention_transpose(const ov::Output<ov::Node>& input, const std::string& name);

/// Repeat KV heads for GQA: [batch, kv_heads, seq, dim] -> [batch, num_heads, seq, dim]
ov::Output<ov::Node> make_repeat_kv(const ov::Output<ov::Node>& kv,
                                    size_t num_heads,
                                    size_t num_kv_heads,
                                    size_t head_dim,
                                    const std::string& name,
                                    const ov::Output<ov::Node>& shared_broadcast_shape = {});

/// Create KV cache read state: Variable + init(zeros) + ReadValue + Gather(beam_idx)
KVCacheReadState make_kv_cache_read(const ov::Output<ov::Node>& batch_source,
                                    const ov::Output<ov::Node>& beam_idx,
                                    size_t num_heads,
                                    size_t head_dim,
                                    const std::string& name,
                                    ov::element::Type precision = ov::element::f32);

/// Create stateful KV cache pattern (ReadValue -> Gather(beam_idx) -> Concat -> Assign)
KVCacheResult make_kv_cache_concat(const ov::Output<ov::Node>& current_kv,
                                   const ov::Output<ov::Node>& batch_source,
                                   const ov::Output<ov::Node>& beam_idx,
                                   size_t num_heads,
                                   size_t head_dim,
                                   const std::string& name,
                                   ov::element::Type precision = ov::element::f32);

/// Compute scaled dot-product attention using native SDPA v13 op
/// When head_dim_for_scale > 0, creates a 5-input SDPA with explicit scale constant
/// (required for embedding model ReConstructEmbeddingModel pattern matching)
ov::Output<ov::Node> make_sdpa(const ov::Output<ov::Node>& q,
                               const ov::Output<ov::Node>& k,
                               const ov::Output<ov::Node>& v,
                               const std::string& name,
                               const ov::Output<ov::Node>& attention_mask = ov::Output<ov::Node>(),
                               size_t head_dim_for_scale = 0);

/// Attention output: transpose back + reshape [batch, seq, hidden] + O projection
ov::Output<ov::Node> make_attention_output(const ov::Output<ov::Node>& sdpa_output,
                                           size_t hidden_size,
                                           const std::string& name,
                                           ov::element::Type precision,
                                           const WeightFn& weight_fn,
                                           const WeightFn& bias_fn = {});

/// Token embedding lookup
ov::Output<ov::Node> make_embedding(const ov::Output<ov::Node>& input_ids,
                                    size_t vocab_size,
                                    size_t hidden_size,
                                    const std::string& name,
                                    ov::element::Type precision = ov::element::f32);

/// LM head (linear projection to vocabulary)
ov::Output<ov::Node> make_lm_head(const ov::Output<ov::Node>& hidden_states,
                                  size_t hidden_size,
                                  size_t vocab_size,
                                  const std::string& name,
                                  ov::element::Type precision = ov::element::f32,
                                  const WeightFn& weight_fn = FP32Weight{});

/// 1D convolution: [batch, in_channels, length] -> [batch, out_channels, out_length]
/// Uses ov::op::v1::Convolution with bias Add.
ov::Output<ov::Node> make_conv1d(const ov::Output<ov::Node>& input,
                                 size_t in_channels,
                                 size_t out_channels,
                                 size_t kernel_size,
                                 size_t stride,
                                 size_t padding,
                                 const std::string& name,
                                 ov::element::Type precision = ov::element::f32);

/// Encoder KV cache (store-only, no concat): ReadValue(init) -> {SDPA, Assign}
/// For cross-attention where encoder KV is computed once and reused across decode steps.
/// No Gather(beam_idx) — encoder KV is identical across beams.
KVCacheResult make_encoder_kv_cache(const ov::Output<ov::Node>& encoder_kv,
                                    size_t num_heads,
                                    size_t head_dim,
                                    const std::string& name,
                                    ov::element::Type precision = ov::element::f32);

// ============================================================================
// Attention functor
// ============================================================================

/// Unified attention mechanism: Q/K/V proj -> reshape -> optional QK-norm ->
/// transpose -> optional RoPE -> optional KV cache -> GQA repeat -> SDPA -> O proj.
/// Handles self-attention, cross-attention, and all cache modes.
struct Attention {
    // Dimensions
    size_t hidden_size, num_heads, num_kv_heads, head_dim;
    ov::element::Type precision;
    WeightFn weight_fn;

    // Per-projection options
    WeightFn bias_fn;  // Optional bias functor for Q/K/V/O projections (empty = no bias)
    NormFn qk_norm;    // Optional QK-norm applied to Q and K after reshape, before RoPE

    // RoPE (empty = no RoPE).  Position IDs are baked into the functor at construction.
    RoPEFn rope_fn;

    // KV cache config (architecture-level, not per-call)
    enum class CacheMode { None, ConcatBeam, StoreOnly };
    std::string cache_infix = ".";  // "." for LLM, ".decoder." / ".encoder." for whisper

    // Whisper layer-0: reuse pre-built key cache Variable to avoid duplicate variable_ids
    std::shared_ptr<ov::op::util::Variable> prebuilt_k_variable;
    ov::Output<ov::Node> prebuilt_k_beam_gather;

    // Mask
    ov::Output<ov::Node> sdpa_mask;

    // Shared GQA broadcast shape (embedding models): when set, always do Unsqueeze→Broadcast→Reshape
    // even for MHA (n_rep=1), using this shared Concat node so all SDPA nodes reference the same shape.
    ov::Output<ov::Node> shared_broadcast_shape;

    // Projection naming (defaults = LLM-style)
    std::string q_proj_name = "self_attn.q_proj";
    std::string k_proj_name = "self_attn.k_proj";
    std::string v_proj_name = "self_attn.v_proj";
    std::string o_proj_name = "self_attn.o_proj";

    LayerResult operator()(const ov::Output<ov::Node>& input,
                           const std::string& prefix,
                           size_t layer_idx = 0,
                           CacheMode cache_mode = CacheMode::None,
                           const ov::Output<ov::Node>& kv_source = {},
                           const ov::Output<ov::Node>& batch_source = {},
                           const ov::Output<ov::Node>& beam_idx = {}) const;
};

// ============================================================================
// Template decoder layer composition
// ============================================================================

/// Whisper decoder layer: 3 sublayers (self-attn, cross-attn, FFN) with norm + residual.
template <typename Norm, typename SelfAttn, typename CrossAttn, typename FFN>
LayerResult make_whisper_decoder_layer(const ov::Output<ov::Node>& input,
                                       const Norm& norm,
                                       const SelfAttn& self_attn,
                                       const CrossAttn& cross_attn,
                                       const FFN& ffn,
                                       const std::string& prefix) {
    // Self-attention: norm -> self_attn -> residual
    auto normed1 = norm(input, prefix + "self_attn_layer_norm");
    auto self_attn_result = self_attn(normed1, prefix);
    auto residual1 = std::make_shared<ov::opset11::Add>(input, self_attn_result.output);
    residual1->set_friendly_name(prefix + "self_attn_residual");

    // Cross-attention: norm -> cross_attn -> residual
    auto normed2 = norm(residual1->output(0), prefix + "encoder_attn_layer_norm");
    auto cross_attn_result = cross_attn(normed2, prefix);
    auto residual2 = std::make_shared<ov::opset11::Add>(residual1, cross_attn_result.output);
    residual2->set_friendly_name(prefix + "cross_attn_residual");

    // FFN: norm -> ffn -> residual
    auto normed3 = norm(residual2->output(0), prefix + "final_layer_norm");
    auto ffn_out = ffn(normed3, prefix + "fc");
    auto residual3 = std::make_shared<ov::opset11::Add>(residual2, ffn_out);
    residual3->set_friendly_name(prefix + "ffn_residual");

    // Merge sinks from self-attn and cross-attn
    std::vector<std::shared_ptr<ov::Node>> sinks;
    sinks.insert(sinks.end(), self_attn_result.sinks.begin(), self_attn_result.sinks.end());
    sinks.insert(sinks.end(), cross_attn_result.sinks.begin(), cross_attn_result.sinks.end());

    return {residual3->output(0), sinks};
}

/// Compose norm + attention + FFN + residual connections into a decoder layer.
/// Extra args are forwarded to the attention functor.
template <typename Norm, typename Attn, typename FFN, typename... Args>
LayerResult make_decoder_layer(const ov::Output<ov::Node>& input,
                               const Norm& norm,
                               const Attn& attention,
                               const FFN& ffn,
                               const std::string& prefix,
                               Args&&... args) {
    // Pre-attention norm + attention + residual
    auto normed1 = norm(input, prefix + "input_layernorm");
    auto attn_result = attention(normed1, prefix, std::forward<Args>(args)...);
    auto residual1 = std::make_shared<ov::opset11::Add>(input, attn_result.output);
    residual1->set_friendly_name(prefix + "attn_residual");

    // Post-attention norm + FFN + residual
    auto normed2 = norm(residual1->output(0), prefix + "post_attention_layernorm");
    auto ffn_out = ffn(normed2, prefix + "mlp");
    auto residual2 = std::make_shared<ov::opset11::Add>(residual1, ffn_out);
    residual2->set_friendly_name(prefix + "ffn_residual");

    return {residual2->output(0), attn_result.sinks};
}

/// BERT-style post-norm layer: attn -> Add(residual) -> Norm -> FFN -> Add(residual) -> Norm.
/// Norm is applied AFTER the residual connection (post-norm), unlike make_decoder_layer (pre-norm).
template <typename Norm, typename Attn, typename FFN, typename... Args>
LayerResult make_post_norm_layer(const ov::Output<ov::Node>& input,
                                 const Norm& norm,
                                 const Attn& attention,
                                 const FFN& ffn,
                                 const std::string& prefix,
                                 Args&&... args) {
    // Attention + residual + norm
    auto attn_result = attention(input, prefix, std::forward<Args>(args)...);
    auto residual1 = std::make_shared<ov::opset11::Add>(input, attn_result.output);
    residual1->set_friendly_name(prefix + "attn_residual");
    auto normed1 = norm(residual1->output(0), prefix + "attention.output.LayerNorm");

    // FFN + residual + norm
    auto ffn_out = ffn(normed1, prefix + "output");
    auto residual2 = std::make_shared<ov::opset11::Add>(normed1, ffn_out);
    residual2->set_friendly_name(prefix + "ffn_residual");
    auto normed2 = norm(residual2->output(0), prefix + "output.LayerNorm");

    return {normed2, attn_result.sinks};
}

// ============================================================================
// LLMConfig
// ============================================================================

/// Configuration for LLM test model building
struct LLMConfig {
    size_t hidden_size = 64;
    size_t num_heads = 4;
    size_t head_dim = 16;
    size_t num_kv_heads = 0;  ///< 0 = MHA (same as num_heads)
    size_t intermediate_size = 256;
    size_t vocab_size = 1000;
    size_t num_layers = 2;

    bool use_kv_cache = true;
    bool use_inputs_embeds = false;  ///< true = inputs_embeds parameter, false = input_ids + embedding
    bool use_lm_head = true;         ///< false = output last_hidden_state (embedding models)
    bool internal_position_ids = false;  ///< true = arange from input shape (no parameter)

    ov::element::Type precision = ov::element::f32;

    /// Functor fields (build_llm fills remaining defaults from scalar params)
    NormFn norm;
    FFNFn ffn;
    RoPEFn rope;  ///< 2-arg: (input, name). Empty = auto HalfRotationRoPE. Set identity lambda to disable.

    /// Position IDs for RoPE. Empty = build_llm auto-creates [batch, seq] Parameter + HalfRotationRoPE.
    /// Provide a custom node (e.g. make_position_ids_3d(), Range subgraph) for non-standard cases.
    /// When pre-building rope, also set this so build_llm tracks the Parameter.
    ov::Output<ov::Node> position_ids;

    WeightFn weight = FP32Weight{};
    WeightFn lm_head_weight = FP32Weight{};
    NormFn qk_norm;  ///< Optional QK-norm forwarded to Attention

    size_t get_kv_heads() const {
        return num_kv_heads == 0 ? num_heads : num_kv_heads;
    }
};

// ============================================================================
// WhisperConfig
// ============================================================================

/// Configuration for Whisper encoder-decoder model building
struct WhisperConfig {
    size_t d_model = 384;
    size_t encoder_layers = 4;
    size_t decoder_layers = 4;
    size_t encoder_attention_heads = 6;
    size_t decoder_attention_heads = 6;
    size_t encoder_ffn_dim = 1536;
    size_t decoder_ffn_dim = 1536;
    size_t vocab_size = 51865;
    size_t num_mel_bins = 80;
    size_t max_source_positions = 1500;
    size_t max_target_positions = 448;
    ov::element::Type precision = ov::element::f32;
    WeightFn weight;

    size_t head_dim() const {
        return d_model / decoder_attention_heads;
    }
};

// ============================================================================
// BERTConfig
// ============================================================================

/// Configuration for BERT-style encoder model building (e.g. Contriever)
struct BERTConfig {
    size_t hidden_size = 768;
    size_t num_heads = 12;
    size_t head_dim = 64;
    size_t intermediate_size = 3072;
    size_t vocab_size = 30522;
    size_t num_layers = 12;
    size_t max_position_embeddings = 512;
    size_t type_vocab_size = 2;
    ov::element::Type precision = ov::element::f32;
    WeightFn weight = FP32Weight{};
    NormFn norm;
    FFNFn ffn;
};

// ============================================================================
// ModelBuilder
// ============================================================================

/// Modular model builder for NPUW testing
///
/// Provides composable building blocks for constructing transformer models.
/// Variant behavior (norm, FFN, RoPE, weights) is controlled via functors
/// stored in LLMConfig, enabling extension without API changes.
///
/// Example with functor blocks:
///   ModelBuilder b;
///   auto input = b.parameter(ov::element::f32, {-1, -1, 64}, "input");
///   RMSNorm norm(64);
///   auto normed = norm(input->output(0), "norm");
///   b.result(normed, "output");
///   auto model = b.build("synthetic_model");
///
/// Or use the convenience method for complete LLMs:
///   auto model = b.build_llm(config);
class ModelBuilder {
public:
    ModelBuilder() = default;

    // ===== Simple test models (backward compatibility) =====
    std::shared_ptr<ov::Model> get_model_with_one_op();
    std::shared_ptr<ov::Model> get_model_without_repeated_blocks();
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks(std::size_t repetitions);
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks();
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks_and_results(
        std::size_t repetitions,
        const std::vector<std::size_t>& block_indices);
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks_and_parameters(
        std::size_t repetitions,
        const std::vector<std::size_t>& block_indices);
    std::shared_ptr<ov::Model> get_model_with_multi_output_repeating_blocks(std::size_t repetitions,
                                                                            bool last_block_has_direct_result);

    // ===== Builder interface =====

    /// Create and track a model parameter
    std::shared_ptr<ov::op::v0::Parameter> parameter(ov::element::Type type,
                                                     const ov::PartialShape& shape,
                                                     const std::string& name);

    /// Create and track a model result
    std::shared_ptr<ov::op::v0::Result> result(const ov::Output<ov::Node>& output, const std::string& name);

    /// Build model from all tracked parameters and results
    std::shared_ptr<ov::Model> build(const std::string& name = "");

    /// Build complete LLM model using configuration
    std::shared_ptr<ov::Model> build_llm(const LLMConfig& config = LLMConfig{});

    /// Build Whisper encoder model
    std::shared_ptr<ov::Model> build_whisper_encoder(const WhisperConfig& config);

    /// Build Whisper decoder model
    std::shared_ptr<ov::Model> build_whisper_decoder(const WhisperConfig& config);

    /// Build BERT-style encoder model (e.g. Contriever)
    std::shared_ptr<ov::Model> build_bert_encoder(const BERTConfig& config);

    // ===== State management =====

    void clear();

private:
    std::shared_ptr<ov::Node> get_block(const std::shared_ptr<ov::Node>& input);
    void set_name(const std::shared_ptr<ov::Node>& node);

    std::vector<std::shared_ptr<ov::Node>> m_nodes;
    std::vector<std::shared_ptr<ov::op::v0::Parameter>> m_parameters;
    std::vector<std::shared_ptr<ov::op::v0::Result>> m_results;
    size_t m_name_idx = 0;
};

}  // namespace npuw
}  // namespace test
}  // namespace ov
