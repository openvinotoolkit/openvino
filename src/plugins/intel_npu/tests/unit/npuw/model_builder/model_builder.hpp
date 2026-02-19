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

struct KVCacheResult {
    ov::Output<ov::Node> concatenated;
    ov::Output<ov::Node> beam_gather;
    std::shared_ptr<ov::Node> assign;
};

struct KVCacheReadState {
    std::shared_ptr<ov::op::util::Variable> variable;
    ov::Output<ov::Node> beam_gather;
};

using WeightFn = std::function<ov::Output<ov::Node>(const std::string&, const ov::Shape&, ov::element::Type)>;
using NormFn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const std::string&)>;
using FFNFn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const std::string&)>;
using RoPEFn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const std::string&)>;
using LayerFn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const std::string&, size_t)>;

/// (projected_k, projected_v, layer_idx) -> (cached_k, cached_v). Empty = no cache.
using KVCacheFn =
    std::function<std::pair<ov::Output<ov::Node>,
                            ov::Output<ov::Node>>(const ov::Output<ov::Node>&, const ov::Output<ov::Node>&, size_t)>;

struct FloatWeight {
    ov::element::Type storage_type;

    FloatWeight(ov::element::Type st = ov::element::f32) : storage_type(st) {}

    ov::Output<ov::Node> operator()(const std::string& name,
                                    const ov::Shape& shape,
                                    ov::element::Type compute_precision) const;
};

using FP32Weight = FloatWeight;
struct FP16Weight : FloatWeight {
    FP16Weight() : FloatWeight(ov::element::f16) {}
};

/// Matches DCOFF SymmNoZP pattern. group_size > 0 = per-group scale.
struct CompressedWeight {
    ov::element::Type storage_type;
    size_t group_size;  ///< 0 = per-channel scale, >0 = per-group scale

    explicit CompressedWeight(ov::element::Type st, size_t gs = 0) : storage_type(st), group_size(gs) {}

    ov::Output<ov::Node> operator()(const std::string& name,
                                    const ov::Shape& shape,
                                    ov::element::Type compute_precision) const;
};

struct INT8Weight : CompressedWeight {
    INT8Weight() : CompressedWeight(ov::element::i8) {}
};

struct INT4Weight : CompressedWeight {
    INT4Weight() : CompressedWeight(ov::element::i4) {}
};

struct INT4GroupWeight : CompressedWeight {
    explicit INT4GroupWeight(size_t gs = 128) : CompressedWeight(ov::element::i4, gs) {}
};

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

/// Position IDs baked in at construction, cos/sin shared across layers.
struct HalfRotationRoPE {
    size_t head_dim;
    ov::Output<ov::Node> cos_freq, sin_freq;

    HalfRotationRoPE(size_t head_dim, ov::element::Type precision, const ov::Output<ov::Node>& position_ids);

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

struct InterleavedRoPE {
    size_t head_dim;
    ov::Output<ov::Node> cos_freq, sin_freq;

    InterleavedRoPE(size_t head_dim, ov::element::Type precision, const ov::Output<ov::Node>& position_ids);

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

/// [batch, seq] position_ids Parameter.
ov::Output<ov::Node> make_position_ids_2d();

/// [3, batch, seq] position_ids Parameter for m-rope. Returns [batch, seq] slice.
ov::Output<ov::Node> make_position_ids_3d();

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

struct GELU {
    size_t hidden_size;
    size_t intermediate_size;
    ov::element::Type precision;
    WeightFn weight_fn;
    WeightFn bias_fn;

    GELU(size_t hs, size_t is, ov::element::Type prec, WeightFn wf, WeightFn bf = {})
        : hidden_size(hs),
          intermediate_size(is),
          precision(prec),
          weight_fn(std::move(wf)),
          bias_fn(std::move(bf)) {}

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

ov::Output<ov::Node> make_linear(const ov::Output<ov::Node>& input,
                                 size_t in_features,
                                 size_t out_features,
                                 const std::string& name,
                                 ov::element::Type precision = ov::element::f32,
                                 const WeightFn& weight_fn = FP32Weight{},
                                 const WeightFn& bias_fn = {});

ov::Output<ov::Node> make_multihead_reshape(const ov::Output<ov::Node>& input,
                                            size_t num_heads,
                                            size_t head_dim,
                                            const std::string& name);

ov::Output<ov::Node> make_attention_transpose(const ov::Output<ov::Node>& input, const std::string& name);

ov::Output<ov::Node> make_repeat_kv(const ov::Output<ov::Node>& kv,
                                    size_t num_heads,
                                    size_t num_kv_heads,
                                    size_t head_dim,
                                    const std::string& name,
                                    const ov::Output<ov::Node>& shared_broadcast_shape = {});

KVCacheReadState make_kv_cache_read(const ov::Output<ov::Node>& batch_source,
                                    const ov::Output<ov::Node>& beam_idx,
                                    size_t num_heads,
                                    size_t head_dim,
                                    const std::string& name,
                                    ov::element::Type precision = ov::element::f32);

KVCacheResult make_kv_cache_concat(const ov::Output<ov::Node>& current_kv,
                                   const ov::Output<ov::Node>& batch_source,
                                   const ov::Output<ov::Node>& beam_idx,
                                   size_t num_heads,
                                   size_t head_dim,
                                   const std::string& name,
                                   ov::element::Type precision = ov::element::f32);

/// head_dim_for_scale > 0 creates 5-input SDPA (needed for ReConstructEmbeddingModel matching)
ov::Output<ov::Node> make_sdpa(const ov::Output<ov::Node>& q,
                               const ov::Output<ov::Node>& k,
                               const ov::Output<ov::Node>& v,
                               const std::string& name,
                               const ov::Output<ov::Node>& attention_mask = ov::Output<ov::Node>(),
                               size_t head_dim_for_scale = 0);

ov::Output<ov::Node> make_attention_output(const ov::Output<ov::Node>& sdpa_output,
                                           size_t hidden_size,
                                           const std::string& name,
                                           ov::element::Type precision,
                                           const WeightFn& weight_fn,
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

/// Store-only KV cache for cross-attention. No beam gather — encoder KV is identical across beams.
KVCacheResult make_encoder_kv_cache(const ov::Output<ov::Node>& encoder_kv,
                                    size_t num_heads,
                                    size_t head_dim,
                                    const std::string& name,
                                    ov::element::Type precision = ov::element::f32);

ov::Output<ov::Node> make_transformer_layers(const ov::Output<ov::Node>& initial,
                                             size_t num_layers,
                                             const std::string& prefix_base,
                                             const LayerFn& layer_fn);

/// Takes pre-projected Q, K, V. Handles reshape, QK-norm, RoPE, KV cache, GQA, SDPA, O proj.
struct Attention {
    size_t hidden_size, num_heads, num_kv_heads, head_dim;
    ov::element::Type precision;
    WeightFn weight_fn;
    WeightFn bias_fn;
    NormFn qk_norm;
    RoPEFn rope_fn;
    KVCacheFn kv_cache_fn;

    ov::Output<ov::Node> sdpa_mask;
    ov::Output<ov::Node> shared_broadcast_shape;

    std::string o_proj_name = "self_attn.o_proj";
    std::string attn_prefix = "self_attn.";

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& q,
                                    const ov::Output<ov::Node>& k,
                                    const ov::Output<ov::Node>& v,
                                    const std::string& prefix,
                                    size_t layer_idx = 0) const;
};

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

/// Unified config for all model types. build_model() dispatches on structural flags.
/// NOTE: weight MUST be declared before lm_head_weight/norm/ffn (C++ member init order).
struct ModelConfig {
    size_t hidden_size = 64;
    size_t num_heads = 4;
    size_t head_dim = 16;
    size_t num_kv_heads = 0;  ///< 0 = MHA
    size_t intermediate_size = 256;
    size_t vocab_size = 1000;
    size_t num_layers = 2;

    bool use_kv_cache = true;
    bool use_inputs_embeds = false;
    bool internal_position_ids = false;

    // Structural flags — build_model() dispatches on these
    bool use_conv_features = false;
    bool use_cross_attention = false;
    bool use_token_type_embedding = false;
    bool pre_norm = true;

    ov::element::Type precision = ov::element::f32;

    WeightFn weight = FP32Weight{};
    WeightFn lm_head_weight;  ///< Truthy = append LM head. Empty = no LM head.
    WeightFn attn_bias;

    NormFn norm;
    FFNFn ffn;
    RoPEFn rope;                        ///< Empty = auto HalfRotationRoPE. Set identity lambda to disable.
    ov::Output<ov::Node> position_ids;  ///< Empty = auto-creates 2D Parameter + HalfRotationRoPE
    NormFn qk_norm;

    // Whisper-specific
    size_t encoder_layers = 0;  ///< 0 = use num_layers
    size_t decoder_layers = 0;  ///< 0 = use num_layers
    size_t num_mel_bins = 80;
    size_t max_source_positions = 1500;
    size_t max_target_positions = 448;

    // BERT/Encoder-specific
    size_t max_position_embeddings = 512;
    size_t type_vocab_size = 2;

    ModelConfig()
        : lm_head_weight(weight),
          norm(LayerNorm(hidden_size, precision)),
          ffn(SwiGLU(hidden_size, intermediate_size, precision, weight)) {}

    size_t get_kv_heads() const {
        return num_kv_heads == 0 ? num_heads : num_kv_heads;
    }

    size_t get_encoder_layers() const {
        return encoder_layers == 0 ? num_layers : encoder_layers;
    }

    size_t get_decoder_layers() const {
        return decoder_layers == 0 ? num_layers : decoder_layers;
    }
};

class ModelBuilder {
public:
    ModelBuilder() = default;

    // Simple test models (backward compat)
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

    std::shared_ptr<ov::op::v0::Parameter> parameter(ov::element::Type type,
                                                     const ov::PartialShape& shape,
                                                     const std::string& name);
    std::shared_ptr<ov::op::v0::Result> result(const ov::Output<ov::Node>& output, const std::string& name);
    std::shared_ptr<ov::Model> build(const std::string& name = "");

    /// Unified entry point. Dispatches on config structural flags.
    std::shared_ptr<ov::Model> build_model(const ModelConfig& config);

    void clear();

private:
    std::shared_ptr<ov::Model> build_llm(const ModelConfig& config);
    std::shared_ptr<ov::Model> build_whisper_encoder(const ModelConfig& config);
    std::shared_ptr<ov::Model> build_whisper_decoder(const ModelConfig& config);
    std::shared_ptr<ov::Model> build_embedding_encoder(const ModelConfig& config);

    /// May auto-create HalfRotationRoPE on config.rope (hence non-const ref).
    ov::Output<ov::Node> setup_position_ids(ModelConfig& config, const ov::Output<ov::Node>& seq_source);

    std::shared_ptr<ov::Model> make_model(const ov::Output<ov::Node>& output,
                                          const std::string& result_name,
                                          const std::string& model_name);

    std::shared_ptr<ov::Node> get_block(const std::shared_ptr<ov::Node>& input);
    void set_name(const std::shared_ptr<ov::Node>& node);

    std::vector<std::shared_ptr<ov::Node>> m_nodes;
    std::vector<std::shared_ptr<ov::op::v0::Parameter>> m_parameters;
    std::vector<std::shared_ptr<ov::op::v0::Result>> m_results;
    ov::SinkVector m_sinks;
    size_t m_name_idx = 0;
};

}  // namespace npuw
}  // namespace test
}  // namespace ov
