// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "model_builder_types.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/util/variable.hpp"

namespace ov {
namespace test {
namespace npuw {

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

/// Store-only KV cache for cross-attention. No beam gather — encoder KV is identical across beams.
KVCacheResult make_encoder_kv_cache(const ov::Output<ov::Node>& encoder_kv,
                                    size_t num_heads,
                                    size_t head_dim,
                                    const std::string& name,
                                    ov::element::Type precision = ov::element::f32);

/// Shared GQA broadcast shape — ReConstructEmbeddingModel requires pointer
/// equality across SDPAs. Build once per model and feed into Attention.
ov::Output<ov::Node> make_shared_gqa_broadcast(const ov::Output<ov::Node>& shape_source,
                                               size_t kv_heads,
                                               size_t num_heads,
                                               size_t head_dim);

/// head_dim_for_scale > 0 creates 5-input SDPA (needed for ReConstructEmbeddingModel matching).
ov::Output<ov::Node> make_sdpa(const ov::Output<ov::Node>& q,
                               const ov::Output<ov::Node>& k,
                               const ov::Output<ov::Node>& v,
                               const std::string& name,
                               const ov::Output<ov::Node>& attention_mask = ov::Output<ov::Node>(),
                               size_t head_dim_for_scale = 0);

/// `attn_dim = num_heads * head_dim`. May differ from `hidden_size` for rectangular projections.
ov::Output<ov::Node> make_attention_output(const ov::Output<ov::Node>& sdpa_output,
                                           size_t hidden_size,
                                           size_t attn_dim,
                                           const std::string& name,
                                           ov::element::Type precision,
                                           const WeightFn& weight_fn,
                                           const WeightFn& bias_fn = {});

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

/// Fixed-size state variable (recurrent/conv states — no sequence growth, optional beam reorder).
struct FixedStateResult {
    std::shared_ptr<ov::op::util::Variable> variable;
    ov::Output<ov::Node> read_value;
};

FixedStateResult make_fixed_state(const ov::Output<ov::Node>& batch_source,
                                  const std::vector<int64_t>& state_dims,
                                  const std::string& name,
                                  ov::element::Type precision = ov::element::f32,
                                  const ov::Output<ov::Node>& beam_idx = {});

/// Post-convolution activation. SiLU matches GatedDeltaNet exports; None matches LFM2 (gated externally).
enum class ConvActivation { None, SiLU };

/// Depthwise conv weight topology. PlainFloat (Const→Convert) matches LFM2; U8Decompress
/// (u8→Convert→Subtract(zp)→Multiply(scale)) matches the Qwen3.5/DCOFF export path.
enum class ConvWeightMode { PlainFloat, U8Decompress };

/// Causal depthwise convolution with sliding-window state.
/// Input/output: [batch, seq, channels]. State: [batch, channels, kernel].
struct CausalConvResult {
    ov::Output<ov::Node> output;       ///< [batch, seq, channels] (post-activation)
    std::shared_ptr<ov::Node> assign;  ///< state update Assign
};

struct CausalConvConfig {
    size_t channels = 0;
    size_t kernel_size = 0;
    ov::element::Type precision = ov::element::f32;
    ConvActivation activation = ConvActivation::SiLU;
    ConvWeightMode weight_mode = ConvWeightMode::U8Decompress;
};

CausalConvResult make_causal_conv(const ov::Output<ov::Node>& input,
                                  const ov::Output<ov::Node>& seq_source,
                                  const ov::Output<ov::Node>& beam_idx,
                                  const std::string& state_name,
                                  const std::string& prefix,
                                  const CausalConvConfig& cfg);

/// Recurrent SSM state via OV Loop. Body implements the GDN delta rule for FuseGDNLoop.
struct RecurrentStateResult {
    ov::Output<ov::Node> output;       ///< [batch, seq, num_heads, value_head_dim]
    std::shared_ptr<ov::Node> assign;  ///< state update Assign
};

/// Per-timestep SSM inputs (all [batch, seq, *]) — grouped so callers can't transpose same-typed args.
struct RecurrentStateInputs {
    ov::Output<ov::Node> query;
    ov::Output<ov::Node> key;
    ov::Output<ov::Node> value;
    ov::Output<ov::Node> gate;
    ov::Output<ov::Node> beta;
};

struct RecurrentStateDims {
    size_t num_heads = 0;
    size_t key_head_dim = 0;
    size_t value_head_dim = 0;
};

RecurrentStateResult make_recurrent_state(const RecurrentStateInputs& inputs,
                                          const RecurrentStateDims& dims,
                                          const ov::Output<ov::Node>& seq_source,
                                          const ov::Output<ov::Node>& beam_idx,
                                          const std::string& state_name,
                                          const std::string& prefix,
                                          ov::element::Type prec);

/// Uniform result for any linear-token mixer. `sinks` already holds the state Assign nodes —
/// the builder appends them with no knowledge of how many a given mixer produces.
struct MixerResult {
    ov::Output<ov::Node> output;
    ov::SinkVector sinks;
};

/// Common runtime plumbing every linear mixer needs, wired once by build_llm.
struct LinearMixer {
    size_t hidden_size = 0;
    ov::element::Type precision = ov::element::f32;
    WeightFn weight_fn;

    // Wired by the builder once for all layers.
    ov::Output<ov::Node> seq_source;
    ov::Output<ov::Node> beam_idx;

    virtual ~LinearMixer() = default;

    virtual MixerResult build(const ov::Output<ov::Node>& input,
                              const std::string& prefix,
                              size_t linear_layer_idx) const = 0;
};

/// GatedDeltaNet (Qwen3.5-style): causal conv + L2-normed Q/K + recurrent SSM + output gating.
struct GatedDeltaNetMixer : LinearMixer {
    size_t num_heads = 0;
    size_t key_head_dim = 0;
    size_t value_head_dim = 0;
    size_t conv_kernel = 4;
    NormFn qk_norm;   ///< empty → L2Norm
    NormFn out_norm;  ///< empty → RMSNorm over value_dim()

    size_t key_dim() const {
        return num_heads * key_head_dim;
    }
    size_t value_dim() const {
        return num_heads * value_head_dim;
    }
    size_t conv_dim() const {
        return 2 * key_dim() + value_dim();
    }

    MixerResult build(const ov::Output<ov::Node>& input,
                      const std::string& prefix,
                      size_t linear_layer_idx) const override;
};

/// LFM2-style gated short convolution (no recurrence, conv state only):
/// in_proj(h → 3*conv_dim) → split(B, C, x) → B*x → causal conv → C*conv → out_proj.
struct ShortConvMixer : LinearMixer {
    size_t conv_dim = 0;  ///< mixer width (LFM2-1.2B: 2048)
    size_t conv_kernel = 3;

    MixerResult build(const ov::Output<ov::Node>& input,
                      const std::string& prefix,
                      size_t linear_layer_idx) const override;
};

/// `mamba_ratio` linear-mixer layers per 1 full-attention layer (0 → empty = pure attention).
std::function<bool(size_t)> make_mamba_schedule(size_t mamba_ratio);

/// Schedule from an explicit attention-layer index list — full attention at those indices,
/// linear everywhere else. Mirrors how LFM2/hybrid configs declare per-layer `layer_types`
/// (e.g. LFM2-1.2B attention at {2, 5, 8, 10, 12, 14}); no closed-form ratio fits.
std::function<bool(size_t)> make_schedule_with_attention_at(std::vector<size_t> attn_layers);

}  // namespace npuw
}  // namespace test
}  // namespace ov
