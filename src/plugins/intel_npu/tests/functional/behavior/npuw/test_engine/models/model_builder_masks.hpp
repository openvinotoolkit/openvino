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

/// Sentinel value used in additive float attention masks for "do not attend" positions.
constexpr float kAttentionMaskPadding = -10000.0f;

/// Padding-only mask: [batch, seq] -> [batch, 1, 1, seq] float (0.0=attend, kAttentionMaskPadding=pad).
ov::Output<ov::Node> make_padding_mask(const ov::Output<ov::Node>& attention_mask,
                                       ov::element::Type prec);

/// Standard causal mask combined with padding. Returns 4D float mask
/// [batch, 1, seq, total_seq] with 0.0=attend, kAttentionMaskPadding=masked.
ov::Output<ov::Node> make_causal_mask(const ov::Output<ov::Node>& seq_source,
                                      const ov::Output<ov::Node>& attention_mask,
                                      ov::element::Type prec);

/// Cache-position-derived position_ids + total_seq_len, used by Whisper-style
/// decoders that anchor the causal mask on Gather(ShapeOf(beam_gather), 2)
/// — root of the CachePositionInput pattern.
struct CachePositionResult {
    ov::Output<ov::Node> position_ids;    ///< [batch, seq]
    ov::Output<ov::Node> total_seq_len;
    ov::Output<ov::Node> seq_len;
    ov::Output<ov::Node> cache_pos_unsq;  ///< [1, seq] (needed for causal mask)
    ov::Output<ov::Node> ids_shape;       ///< ShapeOf(input_ids)
};

CachePositionResult make_cache_position_ids(const ov::Output<ov::Node>& input_ids,
                                            const ov::Output<ov::Node>& kv_cache_beam_gather,
                                            const std::string& prefix);

/// Whisper-style decoder self-attn causal mask. Uses cache_pos based ranges
/// (different shape from make_causal_mask). Includes the structural Slice that
/// AttentionMaskInput (prefill) requires on SDPA input[3].
ov::Output<ov::Node> make_whisper_causal_mask(const CachePositionResult& cache_pos, const std::string& prefix);

}  // namespace npuw
}  // namespace test
}  // namespace ov
