// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

namespace ov {
namespace test {
namespace npuw {

/// FFN activation type
enum class FFNType {
    SWIGLU,  ///< SwiGLU activation with gated linear unit
    GELU     ///< GELU activation
};

/// Normalization type
enum class NormType { LAYER_NORM, RMS_NORM };

/// Rotary Position Embedding type
enum class RoPEType {
    HALF_ROTATION,  ///< Split in half, rotate each half
    INTERLEAVED     ///< Interleave odd/even elements
};

/// Weight storage format for compressed weight patterns
/// INT4/INT8 match DCOFF SymmNoZP pattern: low-bit -> Convert(f16) -> Multiply(f16 scale)
enum class WeightFormat {
    FP32,  ///< f32 weights (default, no compression)
    FP16,  ///< f16 weights with Convert to compute precision
    INT8,  ///< i8 weights -> Convert(f16) -> Multiply(f16 per-channel scale)
    INT4   ///< i4 weights -> Convert(f16) -> Multiply(f16 per-channel scale)
};

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
    bool use_position_ids = true;  ///< Include position_ids parameter and apply RoPE

    FFNType ffn_type = FFNType::SWIGLU;
    NormType norm_type = NormType::LAYER_NORM;
    RoPEType rope_type = RoPEType::HALF_ROTATION;

    WeightFormat weight_format = WeightFormat::FP32;   ///< Default weight format for linear layers
    WeightFormat lm_head_format = WeightFormat::FP32;  ///< LM head weight format (configurable separately)

    bool use_dynamic_shapes = true;
    size_t batch_size = 1;
    size_t seq_len = 1;
    size_t context_len = 128;

    ov::element::Type precision = ov::element::f32;

    size_t get_kv_heads() const {
        return num_kv_heads == 0 ? num_heads : num_kv_heads;
    }
    size_t get_past_len() const {
        return context_len > seq_len ? context_len - seq_len : 0;
    }
};

/// Result from building a decoder layer or attention block
struct LayerResult {
    ov::Output<ov::Node> output;
    std::vector<std::shared_ptr<ov::Node>> sinks;  // Assign nodes for stateful KV cache
};

/// Modular model builder for NPUW testing
///
/// Provides composable building blocks for constructing transformer models.
/// Blocks can be used independently or composed via the builder pattern:
///
///   ModelBuilder b;
///   auto input = b.parameter(ov::element::f32, {-1, -1, 64}, "input");
///   auto normed = b.make_norm(input->output(0), 64, "norm", NormType::RMS_NORM);
///   auto ffn_out = b.make_ffn(normed, 64, 256, "ffn", FFNType::SWIGLU);
///   b.result(ffn_out, "output");
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

    // ===== Atomic building blocks =====

    /// Linear projection (MatMul with weight + optional bias)
    /// Supports compressed weight formats via weight_format parameter
    ov::Output<ov::Node> make_linear(const ov::Output<ov::Node>& input,
                                     size_t out_features,
                                     const std::string& name,
                                     ov::element::Type precision = ov::element::f32,
                                     bool add_bias = false,
                                     WeightFormat weight_format = WeightFormat::FP32);

    /// Normalization (LayerNorm or RMSNorm)
    ov::Output<ov::Node> make_norm(const ov::Output<ov::Node>& input,
                                   size_t hidden_size,
                                   const std::string& name,
                                   NormType type = NormType::LAYER_NORM,
                                   ov::element::Type precision = ov::element::f32,
                                   float eps = 1e-5f);

    // ===== Attention components =====

    /// Reshape for multi-head: [batch, seq, hidden] -> [batch, seq, heads, head_dim]
    ov::Output<ov::Node> make_multihead_reshape(const ov::Output<ov::Node>& input,
                                                size_t num_heads,
                                                size_t head_dim,
                                                const std::string& name);

    /// Transpose for attention: [batch, seq, heads, dim] <-> [batch, heads, seq, dim]
    ov::Output<ov::Node> make_attention_transpose(const ov::Output<ov::Node>& input,
                                                  const std::string& name,
                                                  bool reverse = false);

    /// Repeat KV heads for GQA: [batch, kv_heads, seq, dim] -> [batch, num_heads, seq, dim]
    ov::Output<ov::Node> make_repeat_kv(const ov::Output<ov::Node>& kv,
                                        size_t num_heads,
                                        size_t num_kv_heads,
                                        const std::string& name);

    struct KVCacheResult {
        ov::Output<ov::Node> concatenated;
        std::shared_ptr<ov::Node> assign;  // Sink node for stateful model
    };

    /// Create stateful KV cache pattern (ReadValue -> Gather(beam_idx) -> Concat -> Assign)
    /// batch_source is used to extract the batch dimension (typically input_ids)
    /// beam_idx is needed for beam search reordering (required by StatefulToStateless)
    KVCacheResult make_kv_cache_concat(const ov::Output<ov::Node>& current_kv,
                                       const ov::Output<ov::Node>& batch_source,
                                       const ov::Output<ov::Node>& beam_idx,
                                       size_t num_heads,
                                       size_t head_dim,
                                       const std::string& name,
                                       ov::element::Type precision = ov::element::f32);

    /// Compute scaled dot-product attention using native SDPA v13 op
    /// Required by SDPAToPagedAttention transformation for CPU/GPU backends
    ov::Output<ov::Node> make_attention(const ov::Output<ov::Node>& q,
                                        const ov::Output<ov::Node>& k,
                                        const ov::Output<ov::Node>& v,
                                        size_t head_dim,
                                        const std::string& name,
                                        ov::element::Type precision = ov::element::f32,
                                        const ov::Output<ov::Node>& attention_mask = ov::Output<ov::Node>());

    // ===== Positional encoding =====

    /// Apply Rotary Position Embedding (RoPE)
    /// Input shape: [batch, seq, heads, head_dim] (before attention transpose)
    /// position_ids shape: [batch, seq]
    ov::Output<ov::Node> make_rope(const ov::Output<ov::Node>& input,
                                   const ov::Output<ov::Node>& position_ids,
                                   size_t head_dim,
                                   size_t max_position,
                                   const std::string& name,
                                   RoPEType type = RoPEType::HALF_ROTATION,
                                   ov::element::Type precision = ov::element::f32);

    // ===== FFN =====

    /// Feed-forward network (SwiGLU or GELU)
    ov::Output<ov::Node> make_ffn(const ov::Output<ov::Node>& input,
                                  size_t hidden_size,
                                  size_t intermediate_size,
                                  const std::string& name,
                                  FFNType type = FFNType::SWIGLU,
                                  ov::element::Type precision = ov::element::f32,
                                  WeightFormat weight_format = WeightFormat::FP32);

    // ===== Composite blocks =====

    /// Complete attention block: norm -> Q/K/V proj -> [RoPE] -> transpose -> [KV cache] -> SDPA -> O proj + residual
    /// attention_mask should be pre-transformed to float [batch, 1, seq_len, total_seq] additive mask.
    /// Transform once before the layer loop (not per-layer) for proper NPUW repeating block detection.
    LayerResult make_attention_block(const ov::Output<ov::Node>& input,
                                     const LLMConfig& config,
                                     size_t layer_idx,
                                     const std::string& prefix,
                                     const ov::Output<ov::Node>& position_ids,
                                     const ov::Output<ov::Node>& batch_source,
                                     const ov::Output<ov::Node>& beam_idx,
                                     const ov::Output<ov::Node>& attention_mask);

    /// FFN block: norm -> FFN + residual
    ov::Output<ov::Node> make_ffn_block(const ov::Output<ov::Node>& input,
                                        const LLMConfig& config,
                                        const std::string& prefix);

    /// Complete decoder layer: attention block + FFN block
    LayerResult make_decoder_layer(const ov::Output<ov::Node>& input,
                                   const LLMConfig& config,
                                   size_t layer_idx,
                                   const ov::Output<ov::Node>& position_ids,
                                   const ov::Output<ov::Node>& batch_source,
                                   const ov::Output<ov::Node>& beam_idx,
                                   const ov::Output<ov::Node>& attention_mask);

    // ===== Embedding & head =====

    ov::Output<ov::Node> make_embedding(const ov::Output<ov::Node>& input_ids,
                                        size_t vocab_size,
                                        size_t hidden_size,
                                        const std::string& name,
                                        ov::element::Type precision = ov::element::f32);

    ov::Output<ov::Node> make_lm_head(const ov::Output<ov::Node>& hidden_states,
                                      size_t hidden_size,
                                      size_t vocab_size,
                                      const std::string& name,
                                      ov::element::Type precision = ov::element::f32,
                                      WeightFormat weight_format = WeightFormat::FP32);

    // ===== State management =====

    void clear();
    void track(const std::shared_ptr<ov::Node>& node);

private:
    ov::Output<ov::Node> make_layer_norm(const ov::Output<ov::Node>& input,
                                         size_t hidden_size,
                                         const std::string& name,
                                         ov::element::Type precision,
                                         float eps);
    ov::Output<ov::Node> make_rms_norm(const ov::Output<ov::Node>& input,
                                       size_t hidden_size,
                                       const std::string& name,
                                       ov::element::Type precision,
                                       float eps);
    ov::Output<ov::Node> make_swiglu_ffn(const ov::Output<ov::Node>& input,
                                         size_t hidden_size,
                                         size_t intermediate_size,
                                         const std::string& name,
                                         ov::element::Type precision,
                                         WeightFormat weight_format);
    ov::Output<ov::Node> make_gelu_ffn(const ov::Output<ov::Node>& input,
                                       size_t hidden_size,
                                       size_t intermediate_size,
                                       const std::string& name,
                                       ov::element::Type precision,
                                       WeightFormat weight_format);
    ov::Output<ov::Node> make_half_rotation_rope(const ov::Output<ov::Node>& input,
                                                 const ov::Output<ov::Node>& position_ids,
                                                 size_t head_dim,
                                                 size_t max_position,
                                                 const std::string& name,
                                                 ov::element::Type precision);
    ov::Output<ov::Node> make_interleaved_rope(const ov::Output<ov::Node>& input,
                                               const ov::Output<ov::Node>& position_ids,
                                               size_t head_dim,
                                               size_t max_position,
                                               const std::string& name,
                                               ov::element::Type precision);

    /// Create a weight constant with optional compression (Convert + Scale)
    ov::Output<ov::Node> make_weight(const std::string& name,
                                     size_t rows,
                                     size_t cols,
                                     WeightFormat format,
                                     ov::element::Type compute_precision);

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
