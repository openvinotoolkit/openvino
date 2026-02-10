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
    std::shared_ptr<ov::Node> assign;  // Sink node for stateful model
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
/// call takes (name, rows, cols, compute_precision) -> output
using WeightFn = std::function<ov::Output<ov::Node>(const std::string&, size_t, size_t, ov::element::Type)>;

/// Norm functor: construction captures hidden_size, precision, eps;
/// call takes (input, name) -> output
using NormFn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const std::string&)>;

/// FFN functor: construction captures hidden_size, intermediate_size, precision, weight_fn;
/// call takes (input, name) -> output
using FFNFn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const std::string&)>;

/// RoPE functor: construction captures head_dim, max_position, precision;
/// call takes (input, position_ids, name) -> output
using RoPEFn =
    std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const ov::Output<ov::Node>&, const std::string&)>;

/// PositionIds functor: creates the position_ids tensor (as input parameter or subgraph).
/// For input-based variants, creates a Parameter node (auto-tracked by build_llm).
using PositionIdsFn = std::function<ov::Output<ov::Node>()>;

// ============================================================================
// Weight functors
// ============================================================================

/// f32 weights (no compression)
struct FP32Weight {
    ov::Output<ov::Node> operator()(const std::string& name,
                                    size_t rows,
                                    size_t cols,
                                    ov::element::Type compute_precision) const;
};

/// f16 weights with Convert to compute precision
struct FP16Weight {
    ov::Output<ov::Node> operator()(const std::string& name,
                                    size_t rows,
                                    size_t cols,
                                    ov::element::Type compute_precision) const;
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
                                    size_t rows,
                                    size_t cols,
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

/// Interleaved RoPE: interleave odd/even elements
struct InterleavedRoPE {
    size_t head_dim;
    size_t max_position;
    ov::element::Type precision;

    InterleavedRoPE(size_t hd, size_t mp, ov::element::Type prec = ov::element::f32)
        : head_dim(hd),
          max_position(mp),
          precision(prec) {}

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input,
                                    const ov::Output<ov::Node>& position_ids,
                                    const std::string& name) const;
};

// ============================================================================
// RoPE embedding helper
// ============================================================================

/// Gathered cos/sin embeddings for RoPE, unsqueezed for broadcasting
struct RoPEEmbeddings {
    ov::Output<ov::Node> cos;
    ov::Output<ov::Node> sin;
};

/// Create cos/sin embedding tables, gather by position_ids, and unsqueeze for broadcast.
/// Shared preamble for RoPE: create cos/sin tables, gather, unsqueeze.
RoPEEmbeddings gather_rope_embeddings(size_t head_dim,
                                      size_t max_position,
                                      ov::element::Type precision,
                                      const ov::Output<ov::Node>& position_ids,
                                      const std::string& name);

// ============================================================================
// PositionIds functors
// ============================================================================

/// 2D position_ids as a model input parameter: [batch, seq]
struct Input2DPositionIds {
    ov::Output<ov::Node> operator()() const;
};

/// 3D position_ids as a model input parameter: [3, batch, seq]
/// Returns a 2D slice [batch, seq] (section 0) for downstream RoPE consumption.
/// Matches Qwen2.5-VL's multi-rotary position encoding input shape.
struct Input3DPositionIds {
    ov::Output<ov::Node> operator()() const;
};

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

    GELUFn(size_t hs, size_t is, ov::element::Type prec, WeightFn wf)
        : hidden_size(hs),
          intermediate_size(is),
          precision(prec),
          weight_fn(std::move(wf)) {}

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

// ============================================================================
// Free building-block functions
// ============================================================================

/// Linear projection (MatMul with weight + optional bias)
ov::Output<ov::Node> make_linear(const ov::Output<ov::Node>& input,
                                 size_t in_features,
                                 size_t out_features,
                                 const std::string& name,
                                 ov::element::Type precision = ov::element::f32,
                                 bool add_bias = false,
                                 const WeightFn& weight_fn = FP32Weight{});

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
                                    const std::string& name);

/// Create stateful KV cache pattern (ReadValue -> Gather(beam_idx) -> Concat -> Assign)
KVCacheResult make_kv_cache_concat(const ov::Output<ov::Node>& current_kv,
                                   const ov::Output<ov::Node>& batch_source,
                                   const ov::Output<ov::Node>& beam_idx,
                                   size_t num_heads,
                                   size_t head_dim,
                                   const std::string& name,
                                   ov::element::Type precision = ov::element::f32);

/// Compute scaled dot-product attention using native SDPA v13 op
ov::Output<ov::Node> make_sdpa(const ov::Output<ov::Node>& q,
                               const ov::Output<ov::Node>& k,
                               const ov::Output<ov::Node>& v,
                               const std::string& name,
                               const ov::Output<ov::Node>& attention_mask = ov::Output<ov::Node>());

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

// ============================================================================
// SDPAttention functor
// ============================================================================

/// Full self-attention mechanism: Q/K/V proj -> reshape -> RoPE -> transpose ->
/// KV cache -> GQA repeat -> SDPA -> reshape -> O proj.
/// Does NOT include norm or residual (handled by make_decoder_layer template).
struct SDPAttention {
    size_t hidden_size, num_heads, num_kv_heads, head_dim;
    ov::element::Type precision;
    WeightFn weight_fn;
    RoPEFn rope_fn;  // may be empty (no RoPE)
    bool use_kv_cache;
    ov::Output<ov::Node> position_ids;    // may be empty
    ov::Output<ov::Node> batch_source;    // for KV cache init shape
    ov::Output<ov::Node> beam_idx;        // for beam search reordering
    ov::Output<ov::Node> attention_mask;  // pre-transformed 4D float mask
    size_t layer_idx;

    LayerResult operator()(const ov::Output<ov::Node>& input, const std::string& prefix) const;
};

// ============================================================================
// Template decoder layer composition
// ============================================================================

/// Compose norm + attention + FFN + residual connections into a decoder layer.
/// Works with concrete types (zero-overhead) or std::function types.
template <typename Norm, typename Attention, typename FFN>
LayerResult make_decoder_layer(const ov::Output<ov::Node>& input,
                               const Norm& norm,
                               const Attention& attention,
                               const FFN& ffn,
                               const std::string& prefix) {
    // Pre-attention norm + attention + residual
    auto normed1 = norm(input, prefix + "input_layernorm");
    auto attn_result = attention(normed1, prefix);
    auto residual1 = std::make_shared<ov::opset11::Add>(input, attn_result.output);
    residual1->set_friendly_name(prefix + "attn_residual");

    // Post-attention norm + FFN + residual
    auto normed2 = norm(residual1->output(0), prefix + "post_attention_layernorm");
    auto ffn_out = ffn(normed2, prefix + "mlp");
    auto residual2 = std::make_shared<ov::opset11::Add>(residual1, ffn_out);
    residual2->set_friendly_name(prefix + "ffn_residual");

    return {residual2->output(0), attn_result.sinks};
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

    ov::element::Type precision = ov::element::f32;

    /// Functor fields (build_llm fills remaining defaults from scalar params)
    NormFn norm;
    FFNFn ffn;
    RoPEFn rope;
    PositionIdsFn position_ids = Input2DPositionIds{};  ///< Set to {} for no position_ids/RoPE
    WeightFn weight;
    WeightFn lm_head_weight;

    size_t get_kv_heads() const {
        return num_kv_heads == 0 ? num_heads : num_kv_heads;
    }
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
///   auto model = b.build("my_model");
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
