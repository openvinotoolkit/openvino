// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
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

/// Boolean causal core shared by make_causal_mask / make_causal_mask_boolean /
/// make_sliding_window_mask. Builds absolute query positions (q_range + offset)
/// and key positions, returns the 2D boolean relation (kv <= q, true = attend)
/// plus the two operands the sliding-window variant reuses for its window bound.
struct CausalBool {
    ov::Output<ov::Node> mask;    ///< LessEqual(kv_row, q_col), bool [seq, total_seq]
    ov::Output<ov::Node> q_col;   ///< absolute query positions, [seq, 1]
    ov::Output<ov::Node> kv_row;  ///< key positions, [1, total_seq]
};
CausalBool make_causal_bool(const ov::Output<ov::Node>& seq_source,
                            const ov::Output<ov::Node>& attention_mask,
                            const std::string& prefix);

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
/// boolean_output = true skips the Select-to-float so a bool mask reaches SDPA,
/// exercising NPUW's boolean-mask handling in the Whisper decoder preparation.
ov::Output<ov::Node> make_whisper_causal_mask(const CachePositionResult& cache_pos,
                                              const std::string& prefix,
                                              bool boolean_output = false);

/// Boolean variant of the causal mask: 4D bool (true = attend), no Select-to-float.
/// Element type argument is ignored. Exists to test the NPUW handlers that lift
/// bool SDPA masks to float via Select(mask, 0, -inf). Shape: causal LessEqual +
/// bool padding combined via BitwiseAnd.
ov::Output<ov::Node> make_causal_mask_boolean(const ov::Output<ov::Node>& seq_source,
                                              const ov::Output<ov::Node>& attention_mask,
                                              ov::element::Type /*unused*/);

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
///   - Two SEPARATE Add nodes computing full_ctx_len: one for the Q range,
///     one for the K range — both past_kv_len + seq_len, past FIRST (operand
///     order matters: the matcher's commutative permutation binds its
///     past_kv_len Gather anchor against operand 0 first, and seq_len is
///     also a bare Gather, so past second mis-binds the anchor).
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

/// Sliding window + causal mask, Gemma-4 shape (matches NPUW's
/// Gemma4SlidingMaskMatcher). Same boolean skeleton as the Phi-3 variant —
/// the difference is the Q side: cache_position = Add(Range(0, seq_len),
/// past_kv_len), i.e. the past offset is added AFTER the Range instead of
/// being its start. Window constant is a [1,1,1,1] negative i64 like the
/// real Gemma-4 export. Output is boolean (true = attend).
ov::Output<ov::Node> make_sliding_window_mask_gemma4(const ov::Output<ov::Node>& seq_source,
                                                     const ov::Output<ov::Node>& attention_mask,
                                                     ov::element::Type /*unused*/,
                                                     size_t window_size);

/// Sliding window mask, legacy Phi-3 / transformers 4.51 shape (matches NPUW's
/// OldPhi3SlidingMaskMatcher). INVERTED boolean domain combined via BitwiseOr
/// (true = masked out): Greater(K, Q.T) | LessEqual(K, Q.T - window), with the
/// Q column built via Reshape[-1,1] (not Unsqueeze), the K range double-
/// Converted (i32 → i64 → f32), and a float Q Range. The inverted mask feeds
/// Select(inv, padding, 0) so the SDPA input is a float mask of type `prec`.
ov::Output<ov::Node> make_sliding_window_mask_phi3_legacy(const ov::Output<ov::Node>& seq_source,
                                                          const ov::Output<ov::Node>& attention_mask,
                                                          ov::element::Type prec,
                                                          size_t window_size);

/// Modify a base mask so image tokens (token_type_ids==1) get bidirectional
/// attention among themselves. Used for Gemma 3 VLM. base_mask shape is
/// [batch, 1, seq, total_seq], float (attend = Select to 0.0/prec) or boolean
/// (attend = Select to true, e.g. on top of make_sliding_window_mask_phi3).
/// token_type_ids is [batch, total_seq] i64. With KV cache, seq < total_seq —
/// Q-side token types are the LAST seq_len entries.
ov::Output<ov::Node> make_vlm_bidirectional_modifier(const ov::Output<ov::Node>& base_mask,
                                                     const ov::Output<ov::Node>& token_type_ids,
                                                     const ov::Output<ov::Node>& seq_source,
                                                     ov::element::Type prec);

}  // namespace npuw
}  // namespace test
}  // namespace ov
