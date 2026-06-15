// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace test {
namespace npuw {

/// Position IDs baked in at construction, cos/sin shared across layers.
/// shape_source provides batch dim for inv_freq Broadcast (matches NPUW RopeCache pattern).
/// Defaults to position_ids when not specified.
struct HalfRotationRoPE {
    size_t head_dim;
    ov::Output<ov::Node> cos_freq, sin_freq;

    HalfRotationRoPE(size_t head_dim,
                     ov::element::Type precision,
                     const ov::Output<ov::Node>& position_ids,
                     const ov::Output<ov::Node>& shape_source = {});

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
