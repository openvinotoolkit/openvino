// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "model_builder_attention.hpp"
#include "model_builder_ffn.hpp"
#include "model_builder_masks.hpp"
#include "model_builder_norm.hpp"
#include "model_builder_rope.hpp"
#include "model_builder_types.hpp"
#include "model_builder_weights.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset11.hpp"

namespace ov {
namespace test {
namespace npuw {

ov::Output<ov::Node> make_linear(const ov::Output<ov::Node>& input,
                                 size_t in_features,
                                 size_t out_features,
                                 const std::string& name,
                                 ov::element::Type precision = ov::element::f32,
                                 const WeightFn& weight_fn = FP32Weight{},
                                 const WeightFn& bias_fn = {});

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
                                  const WeightFn& weight_fn = FP32Weight{});

ov::Output<ov::Node> make_conv1d(const ov::Output<ov::Node>& input,
                                 size_t in_channels,
                                 size_t out_channels,
                                 size_t kernel_size,
                                 size_t stride,
                                 size_t padding,
                                 const std::string& name,
                                 ov::element::Type precision = ov::element::f32);

ov::Output<ov::Node> make_transformer_layers(const ov::Output<ov::Node>& initial,
                                             size_t num_layers,
                                             const std::string& prefix_base,
                                             const LayerFn& layer_fn);

template <typename Norm, typename SelfAttn, typename CrossAttn, typename FFN>
ov::Output<ov::Node> make_cross_attn_decoder_layer(const ov::Output<ov::Node>& input,
                                                   const Norm& norm,
                                                   const SelfAttn& self_attn,
                                                   const CrossAttn& cross_attn,
                                                   const FFN& ffn,
                                                   const std::string& prefix) {
    auto normed1 = norm(input, prefix + "self_attn_layer_norm");
    auto self_attn_out = self_attn(normed1, prefix);
    auto residual1 = std::make_shared<ov::opset11::Add>(input, self_attn_out);
    residual1->set_friendly_name(prefix + "self_attn_residual");

    auto normed2 = norm(residual1->output(0), prefix + "encoder_attn_layer_norm");
    auto cross_attn_out = cross_attn(normed2, prefix);
    auto residual2 = std::make_shared<ov::opset11::Add>(residual1, cross_attn_out);
    residual2->set_friendly_name(prefix + "cross_attn_residual");

    auto normed3 = norm(residual2->output(0), prefix + "final_layer_norm");
    auto ffn_out = ffn(normed3, prefix + "fc");
    auto residual3 = std::make_shared<ov::opset11::Add>(residual2, ffn_out);
    residual3->set_friendly_name(prefix + "ffn_residual");

    return residual3->output(0);
}

template <typename Norm, typename Attn, typename FFN, typename... Args>
ov::Output<ov::Node> make_pre_norm_layer(const ov::Output<ov::Node>& input,
                                         const Norm& norm,
                                         const Attn& attention,
                                         const FFN& ffn,
                                         const std::string& prefix,
                                         Args&&... args) {
    auto normed1 = norm(input, prefix + "input_layernorm");
    auto attn_out = attention(normed1, prefix, std::forward<Args>(args)...);
    auto residual1 = std::make_shared<ov::opset11::Add>(input, attn_out);
    residual1->set_friendly_name(prefix + "attn_residual");

    auto normed2 = norm(residual1->output(0), prefix + "post_attention_layernorm");
    auto ffn_out = ffn(normed2, prefix + "mlp");
    auto residual2 = std::make_shared<ov::opset11::Add>(residual1, ffn_out);
    residual2->set_friendly_name(prefix + "ffn_residual");

    return residual2->output(0);
}

template <typename Norm, typename Attn, typename FFN, typename... Args>
ov::Output<ov::Node> make_post_norm_layer(const ov::Output<ov::Node>& input,
                                          const Norm& norm,
                                          const Attn& attention,
                                          const FFN& ffn,
                                          const std::string& prefix,
                                          Args&&... args) {
    auto attn_out = attention(input, prefix, std::forward<Args>(args)...);
    auto residual1 = std::make_shared<ov::opset11::Add>(input, attn_out);
    residual1->set_friendly_name(prefix + "attn_residual");
    auto normed1 = norm(residual1->output(0), prefix + "attention.output.LayerNorm");

    auto ffn_out = ffn(normed1, prefix + "output");
    auto residual2 = std::make_shared<ov::opset11::Add>(normed1, ffn_out);
    residual2->set_friendly_name(prefix + "ffn_residual");
    auto normed2 = norm(residual2->output(0), prefix + "output.LayerNorm");

    return normed2;
}

struct BaseModelConfig {
    // Common parameters
    size_t hidden_size = 64;
    size_t num_heads = 4;
    size_t head_dim = 16;
    size_t num_kv_heads = 0;  ///< 0 = MHA
    size_t intermediate_size = 256;
    size_t vocab_size = 1000;
    size_t num_layers = 10;

    ov::element::Type precision = ov::element::f32;

    WeightFn weight = FP32Weight{};
    WeightFn lm_head_weight;  ///< Truthy = append LM head. Empty = no LM head.
    WeightFn attn_bias;

    NormFn norm;
    FFNFn ffn;
    RoPEFn rope;                        ///< Empty = auto HalfRotationRoPE. Set identity lambda to disable.
    ov::Output<ov::Node> position_ids;  ///< Empty = auto-creates 2D Parameter + HalfRotationRoPE
    NormFn qk_norm;

    BaseModelConfig() : lm_head_weight(weight) {}

    virtual ~BaseModelConfig() = default;

    size_t get_kv_heads() const {
        return num_kv_heads == 0 ? num_heads : num_kv_heads;
    }
};

struct LLMConfig : public BaseModelConfig {
    bool use_kv_cache = true;
    bool use_inputs_embeds = false;
    bool internal_position_ids = false;  ///< embedding model
    bool pre_norm = true;
    bool force_gqa_broadcast = false;  ///< force 5-input SDPA (needed for SDPA isolation pattern matching)

    // MoE configuration (num_experts=0 means dense, no MoE)
    size_t num_experts = 0;           ///< Total experts. 0 = dense model.
    size_t num_experts_per_tok = 0;   ///< Top-K. 0 = default to 2.
    size_t moe_intermediate_size = 0; ///< Expert FFN intermediate size. 0 = use intermediate_size.
};

struct WhisperConfig : public BaseModelConfig {
    size_t encoder_layers = 0;
    size_t decoder_layers = 0;
    size_t num_mel_bins = 80;
    size_t max_source_positions = 1500;
    size_t max_target_positions = 448;

    size_t get_encoder_layers() const {
        return encoder_layers == 0 ? num_layers : encoder_layers;
    }
    size_t get_decoder_layers() const {
        return decoder_layers == 0 ? num_layers : decoder_layers;
    }
    /// Encoder output sequence length after Conv1D preprocessing (stride=2 on 2*max_source_positions).
    size_t get_encoder_seq_len() const {
        return max_source_positions;
    }
};

struct BertConfig : public BaseModelConfig {
    size_t max_position_embeddings = 512;
    size_t type_vocab_size = 2;
};

class ModelBuilder {
public:
    ModelBuilder() = default;

    // Simple test models (backward compat)
    std::shared_ptr<ov::Model> get_model_with_one_op();
    std::shared_ptr<ov::Model> get_model_without_repeated_blocks();
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks(std::size_t repetitions);
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks();
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks_with_weightless_cache(std::size_t repetitions);
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks_with_weightless_cache();
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks_and_results(
        std::size_t repetitions,
        const std::vector<std::size_t>& block_indices);
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks_and_parameters(
        std::size_t repetitions,
        const std::vector<std::size_t>& block_indices);
    // Builds a model with N repeated blocks using a 4-op structure
    // (Add→Relu→Multiply→Relu) where both Relu nodes share the same metadesc.
    // "Head" blocks additionally expose their interior Relu via a cross-group MatMul.
    // Because the interior and boundary Relu share the same metadesc, ALL blocks stay
    // in one repeated-block family regardless of head/non-head status, allowing
    // isRegularCrossGroupConsumerCase to detect the per-bank connectivity asymmetry.
    std::shared_ptr<ov::Model> get_model_with_kv_sharing_repeated_blocks(
        std::size_t repetitions,
        const std::vector<std::size_t>& head_block_indices);
    std::shared_ptr<ov::Model> get_model_with_multi_output_repeating_blocks(std::size_t repetitions,
                                                                            bool last_block_has_direct_result);

    std::shared_ptr<ov::op::v0::Parameter> parameter(ov::element::Type type,
                                                     const ov::PartialShape& shape,
                                                     const std::string& name);

    std::shared_ptr<ov::Model> build_llm(const LLMConfig& config);
    std::shared_ptr<ov::Model> build_whisper_encoder(const WhisperConfig& config);
    std::shared_ptr<ov::Model> build_whisper_decoder(const WhisperConfig& config);
    std::shared_ptr<ov::Model> build_embedding_encoder(const BertConfig& config);

    void clear();

private:
    /// May auto-create HalfRotationRoPE on config.rope (hence non-const ref).
    ov::Output<ov::Node> setup_position_ids(LLMConfig& config, const ov::Output<ov::Node>& seq_source);

    std::shared_ptr<ov::Model> make_model(const ov::Output<ov::Node>& output,
                                          const std::string& result_name,
                                          const std::string& model_name);

    std::shared_ptr<ov::Node> get_block(const std::shared_ptr<ov::Node>& input);
    void set_name(const std::shared_ptr<ov::Node>& node);

    std::vector<std::shared_ptr<ov::Node>> m_nodes;
    ov::SinkVector m_sinks;
    size_t m_name_idx = 0;
};

}  // namespace npuw
}  // namespace test
}  // namespace ov
