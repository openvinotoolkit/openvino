// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "model_builder_types.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

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

ov::Output<ov::Node> make_attention_output(const ov::Output<ov::Node>& sdpa_output,
                                           size_t hidden_size,
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

}  // namespace npuw
}  // namespace test
}  // namespace ov
