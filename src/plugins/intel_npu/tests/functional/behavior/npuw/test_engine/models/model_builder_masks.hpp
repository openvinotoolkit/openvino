// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

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

/// Sliding window + causal mask (transformers >= 5 / Gemma-4 style):
/// Range(0, seq_len)+Add(offset), single Unsqueeze, Subtract for lower bound,
/// LogicalAnd combine, Select to float. Returns 4D float mask.
ov::Output<ov::Node> make_sliding_window_mask(const ov::Output<ov::Node>& seq_source,
                                              const ov::Output<ov::Node>& attention_mask,
                                              ov::element::Type prec,
                                              size_t window_size);

/// Sliding window + causal mask, Phi-3 / older transformers (~4.53) shape.
/// Mirrors Phi3SlidingMaskMatcher pattern — produces a 4D boolean mask
/// (true = attend) that Phi3SlidingMaskMatcher can match-and-rewrite.
///
/// Pattern features (all matter for the matcher):
///   - past_kv_len obtained via Gather (anchored on a Gather node + optional Squeeze).
///   - Two SEPARATE Add nodes computing full_ctx_len: one for Q range
///     (past_kv_len + seq_len, operand order matters), one for K range
///     (seq_len + past_kv_len).
///   - Q range = Range(past_kv_len, full_ctx_len_q) — absolute positions.
///   - K range = Range(0, full_ctx_len_k).
///   - 3x Unsqueeze chain on each range (axes are "any" in matcher).
///   - Lower bound = Add(q_col, neg_window_size_const) — NOT Subtract; constant is negative.
///   - Combine: BitwiseAnd(any_bool, sliding_bool) -> BitwiseAnd(_, causal_bool).
///   - Output is boolean. SDPA accepts boolean masks (true = attend).
///
/// Element type argument is ignored (output is always boolean) — the signature
/// matches SlidingMaskFn so it can be plugged into LLMConfig::sliding_mask_fn.
ov::Output<ov::Node> make_sliding_window_mask_phi3(const ov::Output<ov::Node>& seq_source,
                                                   const ov::Output<ov::Node>& attention_mask,
                                                   ov::element::Type /*unused*/,
                                                   size_t window_size);

/// Modify a float base mask so image tokens (token_type_ids==1) get bidirectional
/// attention among themselves. Used for Gemma 3 VLM. base_mask shape is
/// [batch, 1, seq, total_seq]. token_type_ids is [batch, total_seq] i64. With
/// KV cache, seq < total_seq — Q-side token types are the LAST seq_len entries.
ov::Output<ov::Node> make_vlm_bidirectional_modifier(const ov::Output<ov::Node>& base_mask,
                                                     const ov::Output<ov::Node>& token_type_ids,
                                                     const ov::Output<ov::Node>& seq_source,
                                                     ov::element::Type prec);

}  // namespace npuw
}  // namespace test
}  // namespace ov
