// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace test {
namespace npuw {

/// Phi-style LongRoPE inverse frequencies. When enabled, the inv_freq Constant of the
/// RoPE chain is replaced by
///   max(position_ids) + 1 <= context_limit ? inv_freq_short : inv_freq_long
/// which is what NPUW's LongRopePatternPhi matches and RopeCache rewrites into the
/// host-fed npuw_lr_cos / npuw_lr_sin model inputs.
struct LongRopeSpec {
    std::vector<float> inv_freq_short;  ///< head_dim / 2 entries
    std::vector<float> inv_freq_long;   ///< head_dim / 2 entries
    int64_t context_limit = 0;          ///< original_max_position_embeddings; 0 = plain RoPE

    bool enabled() const {
        return context_limit > 0 && !inv_freq_short.empty();
    }
};

/// Position IDs baked in at construction, cos/sin shared across layers.
/// shape_source provides batch dim for inv_freq Broadcast (matches NPUW RopeCache pattern).
/// Defaults to position_ids when not specified.
struct HalfRotationRoPE {
    size_t head_dim;
    ov::Output<ov::Node> cos_freq, sin_freq;

    HalfRotationRoPE(size_t head_dim,
                     ov::element::Type precision,
                     const ov::Output<ov::Node>& position_ids,
                     const ov::Output<ov::Node>& shape_source = {},
                     const LongRopeSpec& longrope = {});

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

struct InterleavedRoPE {
    size_t head_dim;
    ov::Output<ov::Node> cos_freq, sin_freq;

    InterleavedRoPE(size_t head_dim,
                    ov::element::Type precision,
                    const ov::Output<ov::Node>& position_ids,
                    const ov::Output<ov::Node>& shape_source = {});

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

/// Partial RoPE (Qwen3.5-style): only the first rotary_dim of each head is rotated
/// (half-rotation), the remaining head_dim - rotary_dim passes through, then both are
/// re-concatenated. cos/sin are built over rotary_dim.
struct PartialRotationRoPE {
    size_t head_dim;
    size_t rotary_dim;
    HalfRotationRoPE inner;  ///< rotation over the rotary_dim prefix

    PartialRotationRoPE(size_t head_dim,
                        size_t rotary_dim,
                        ov::element::Type precision,
                        const ov::Output<ov::Node>& position_ids,
                        const ov::Output<ov::Node>& shape_source = {});

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

/// [batch, seq] position_ids Parameter.
ov::Output<ov::Node> make_position_ids_2d();

/// [3, batch, seq] position_ids Parameter for m-rope. Returns [batch, seq] slice.
ov::Output<ov::Node> make_position_ids_3d();

/// Learned absolute positional embedding lookup (no RoPE — used by Whisper).
/// Adds a Gather'd row from a per-position embedding table to the token embeddings.
ov::Output<ov::Node> make_learned_positional_embedding(const ov::Output<ov::Node>& token_embed,
                                                       const ov::Output<ov::Node>& position_ids,
                                                       size_t max_target_positions,
                                                       size_t hidden_size,
                                                       ov::element::Type precision,
                                                       const std::string& prefix);

}  // namespace npuw
}  // namespace test
}  // namespace ov
