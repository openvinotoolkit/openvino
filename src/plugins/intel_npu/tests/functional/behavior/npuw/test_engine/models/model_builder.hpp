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

/// Decompression pattern for CompressedWeight, matching DCOFF recognition.
///
/// After NPUW partitioning, Constants become Parameters.  The patterns below
/// describe the graph that DCOFF will see *before* partitioning transforms it.
///
///   Pattern         | Chain (f16/f32 = decomp type)             | DCOFF class matched
///   ----------------+-------------------------------------------+------------------------------
///   SYMM_NO_ZP      | Cvt(f16) → Mul(f16 scale) [→ Reshape]    | Reshape3 / Reshape4
///   SYMM_NO_ZP_F32  | Cvt(f32) → Mul(f32 scale) [→ Reshape]    | SymmNoZP::MatMul / Reshape4
///   SYMM_ZP         | Cvt(f16) → Sub(Const u4→Cvt f16) → Mul   | Reshape1 / Convert1
///   GPTQ            | Cvt(f32) → Sub(Const f32) → Mul(f32)     | Reshape2
///   ASYMM_ZP        | Cvt(f16) → Sub(varying u4→Cvt f16) → Mul | AsymmZP::Reshape
enum class DCOffPattern {
    SYMM_NO_ZP,      ///< f16 chain, no zero point.  i4/i8/u4 storage.
    SYMM_NO_ZP_F32,  ///< f32 chain, no zero point.  i4/i8/nf4 storage.
    SYMM_ZP,         ///< f16 chain, uniform u4 zero point (Constant after partitioning).  u4 storage.
    GPTQ,            ///< f32 chain, uniform f32 zero point (no Convert on ZP).  u4 storage.
    ASYMM_ZP,        ///< f16 chain, per-layer varying u4 zero point (Parameter after partitioning).  u4 storage.
};

/// Compressed (quantized) weight with configurable DCOFF decompression pattern.
/// group_size > 0 = per-group scale (3D weight → decompress → Reshape 2D).
/// group_size = 0 = per-channel scale (2D weight, no Reshape).
/// Note: GPTQ and ASYMM_ZP require group_size > 0 (no per-channel DCOFF pass exists).
struct CompressedWeight {
    ov::element::Type storage_type;
    size_t group_size;     ///< 0 = per-channel scale, >0 = per-group scale
    DCOffPattern pattern;  ///< Decompression pattern to generate

    explicit CompressedWeight(ov::element::Type st, size_t gs = 0, DCOffPattern pat = DCOffPattern::SYMM_NO_ZP)
        : storage_type(st),
          group_size(gs),
          pattern(pat) {}

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
/// shape_source provides batch dim for inv_freq Broadcast (matches NPUW RopeCache pattern).
/// Defaults to position_ids when not specified.
struct HalfRotationRoPE {
    size_t head_dim;
    ov::Output<ov::Node> cos_freq, sin_freq;

    HalfRotationRoPE(size_t head_dim, ov::element::Type precision,
                     const ov::Output<ov::Node>& position_ids,
                     const ov::Output<ov::Node>& shape_source = {});

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

struct InterleavedRoPE {
    size_t head_dim;
    ov::Output<ov::Node> cos_freq, sin_freq;

    InterleavedRoPE(size_t head_dim, ov::element::Type precision,
                    const ov::Output<ov::Node>& position_ids,
                    const ov::Output<ov::Node>& shape_source = {});

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

/// GPT-OSS style batched MoE FFN matching NPUW's GPTOSSExpert + GPTOSSRouter patterns.
/// All experts compute on all tokens via Tile + 3D batched MatMul (no NonZero).
/// Weight function must produce a Multiply→Convert→MatMul chain (default: i4 CompressedWeight)
/// for the isolation patterns to match.  Shared constants across layers enable repeating
/// block detection.  Conforms to FFNFn for drop-in use in transformer layer templates.
struct MoEFFN {
    size_t hidden_size, intermediate_size, num_experts, num_experts_per_tok;
    ov::element::Type precision;
    WeightFn weight_fn;

    /// Default weight_fn: CompressedWeight{i4, 0, SYMM_NO_ZP}.
    MoEFFN(size_t hs, size_t is, size_t ne, size_t k, ov::element::Type prec, WeightFn wf = {});

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;

private:
    // Shared across layers for matchRepeatedSubgraphs (created once in ctor)
    std::shared_ptr<ov::Node> tile_repeats, topk_k_const;
    std::shared_ptr<ov::Node> slice_step, slice_axis2, slice_start_0, slice_stop_is, slice_start_is, slice_stop_2is;
    std::shared_ptr<ov::Node> min_const, swish_beta, clamp_add_zero;
    std::shared_ptr<ov::Node> sl_start, sl_step_r, sl_axes, scatter_axis;
    std::shared_ptr<ov::Node> tp_order, unsq_axis, reduce_axis;
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

    /// Convenience: project Q/K/V from input (and optionally kv_input for K/V), then attend.
    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input,
                                    const ov::Output<ov::Node>& kv_input,
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

    // MoE configuration (num_experts=0 means dense, no MoE)
    size_t num_experts = 0;           ///< Total experts. 0 = dense model.
    size_t num_experts_per_tok = 0;   ///< Top-K. 0 = default to 2.
    size_t moe_intermediate_size = 0; ///< Expert FFN intermediate size. 0 = use intermediate_size.

    size_t sliding_window_size = 0;      ///< 0 = no sliding window. >0 = window size (Phi-3, Gemma 2/3)
    bool alternating_attention = false;  ///< false = all layers same mask. true = even=sliding, odd=full (Gemma 2)
    bool use_token_type_ids = false;     ///< Gemma 3 VLM: token_type_ids param (0=text/causal, 1=image/bidir)
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
