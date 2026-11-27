// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API PositionIDsReplacer;
class TRANSFORMATIONS_API PositionIDsReplacerQwen;
class TRANSFORMATIONS_API PositionIDsReplacerLFM2;
class TRANSFORMATIONS_API PositionIDsReplacerCodeGen2;
class TRANSFORMATIONS_API EliminateDropBatch;
class TRANSFORMATIONS_API RoPEUnsqueezeAxisReplacer;
class TRANSFORMATIONS_API PositionIDsReplacerGranite;
class TRANSFORMATIONS_API PositionIDsReplacerQwen3_5;

}  // namespace pass
}  // namespace ov

class ov::pass::PositionIDsReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PositionIDsReplacer");
    explicit PositionIDsReplacer(const Output<Node>& position_ids);
};

/**
 * @brief Qwen model expects data processing in order, the "position ids" input is detached and
 * is not explicitly used in the model. The model uses implicitly defined "position ids" based
 * on the past KV cache size.
 *
 * To use this model in Continuous batching mode, we need to apply position_ids and
 * use the corresponding rotary_emb_cos/rotary_emb_sin.
 * For this, we replace
 *      rotary_emb_cos/rotary_emb_sin -> Slice -> Slice
 * With
 *      rotary_emb_cos/rotary_emb_sin -> Gather(by position_ids)
 * Which enables applying RoPE for each token independently of their order in the input tensor.
 */
class ov::pass::PositionIDsReplacerQwen : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PositionIDsReplacerQwen");
    explicit PositionIDsReplacerQwen(const Output<Node>& position_ids);
};

/**
 * @brief Codegen2 model doesn't use the position_ids input explicitly.
 * Instead, the model infers them from the max_context_len value by generating
 * a range from 0 to max_context_len, applying RoPE and only then Slicing the
 * last token which is not correct in case of 0th iteration (prompt iteration)
 * when values for the entire sequence need to be sliced.
 *
 * We change from this:
 *
 *  ┌─────┐
 *  │Range│
 *  └──┬──┘
 *     │
 *  ┌──┴──┐     ┌─────────┐    ┌─────────┐
 *  │RoPE │     │  Start  │    │   End   │
 *  │Block│     │(prev.seq|    │(cur.seq │
 *  └──┬──┘     │ len)    │    │   len)  │
 *     │        └────┬────┘    └────┬────┘
 *  ┌──┴──┐──────────┘              │
 *  |Slice├─────────────────────────┘
 *  └─────┘
 *
 * To this to Gather by position_ids
 *
 *  ┌─────┐
 *  │Range│
 *  └──┬──┘
 *     │
 *  ┌──┴──┐
 *  │RoPE │
 *  │Block│    ┌──────────────┐
 *  └──┬──┘    │ position_ids │
 *     │       └──────┬───────┘
 *  ┌──┴───┐          │
 *  │Gather├──────────┘
 *  └──────┘
 */

class ov::pass::PositionIDsReplacerCodeGen2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PositionIDsReplacerCodeGen2");
    explicit PositionIDsReplacerCodeGen2(const std::shared_ptr<ov::op::v0::Parameter>& position_ids);
};

/**
 * @brief LFM2-style models compute RoPE positions from an internal arange (aten::arange) rather
 * than from the explicit position_ids input. This transformation replaces that arange output with
 * position_ids so that Paged Attention can serve tokens in arbitrary order.
 */
class ov::pass::PositionIDsReplacerLFM2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PositionIDsReplacerLFM2");
    explicit PositionIDsReplacerLFM2(const Output<Node>& position_ids);
};

/**
 * @brief After SDPAToPagedAttention flattens position_ids, the batch dimension no longer exists, so an
 * aten::select(dim=0, index=0) that dropped it (lowered to Gather(index=0, axis=0)) becomes redundant.
 *
 * This transformation detects Parameter(name == "position_ids") -> Unsqueeze(optional) -> Convert(optional) ->
 * Gather(index=0, axis=0) and replaces the Gather with a Reshape(-1), which flattens whatever shape is produced
 * by the optional Unsqueeze/Convert back to a 1D tensor. Since position_ids no longer carries a real batch
 * dimension to select from, this is equivalent to the original select but no longer depends on a batch axis
 * being present.
 *
 * We change from this:                        to this:
 *
 *  ┌──────────────┐                      ┌──────────────┐
 *  │ position_ids │                      │ position_ids │
 *  └──────┬───────┘                      └──────┬───────┘
 *         │                                     │
 *  ┌──────┴──────┐                       ┌──────┴──────┐
 *  │  Unsqueeze  │ (optional)            │  Unsqueeze  │ (optional)
 *  └──────┬──────┘                       └──────┬──────┘
 *         │                                     │
 *  ┌──────┴──────┐                       ┌──────┴──────┐
 *  │   Convert   │ (optional)            │   Convert   │ (optional)
 *  └──────┬──────┘                       └──────┬──────┘
 *         │                                     │
 *  ┌──────┴──────┐                       ┌──────┴──────┐
 *  │Gather(idx=0,│                       │   Reshape   │
 *  │   axis=0)   │                       │    (-1)     │
 *  └─────────────┘                       └─────────────┘
 */
class ov::pass::EliminateDropBatch : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("EliminateDropBatch");
    EliminateDropBatch();
};

/**
 * @brief Some models compute RoPE manually with an explicit outer product (position * inv_freq) followed by
 * Cos/Sin instead of using a fused rotary embedding op, producing cos/sin values in the original
 * [batch=1, tokens, ...] layout via a trailing Unsqueeze(axis=0). PagedAttention's Q/K arrive with tokens
 * flattened into the leading axis instead, so this transformation detects that RoPE outer-product tail
 * (MatMul(inv_freq) -> Cos/Sin -> Multiply(scale) -> Broadcast(optional) -> Unsqueeze(axis=0)) and
 * rewrites the trailing Unsqueeze's axis from 0 to 1, moving the flattened-tokens axis to index 0 to match
 * the layout Q/K arrive in. This is independent of how the per-token positions were derived (works whether
 * or not EliminateDropBatch has already collapsed a batch-drop select feeding into the outer product).
 *
 * We change from this:            to this:
 *
 *  ┌───────┐                ┌───────┐
 *  │ MatMul│                │ MatMul│  (outer product with inv_freq)
 *  └───┬───┘                └───┬───┘
 *      │                        │
 *  ┌───┴───┐                ┌───┴───┐
 *  │Cos/Sin│                │Cos/Sin│
 *  └───┬───┘                └───┬───┘
 *      │                        │
 *  ┌───┴────┐               ┌───┴────┐
 *  │Multiply│ (scale)       │Multiply│ (scale)
 *  └───┬────┘               └───┬────┘
 *      │                        │
 *  ┌───┴─────┐              ┌───┴─────┐
 *  │Broadcast│ (optional)   │Broadcast│ (optional)
 *  └───┬─────┘              └───┬─────┘
 *      │                        │
 *  ┌───┴──────┐             ┌───┴──────┐
 *  │Unsqueeze │             │Unsqueeze │
 *  │ (axis=0) │             │ (axis=1) │
 *  └──────────┘             └──────────┘
 */
class ov::pass::RoPEUnsqueezeAxisReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEUnsqueezeAxisReplacer");
    RoPEUnsqueezeAxisReplacer();
};

class ov::pass::PositionIDsReplacerGranite : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PositionIDsReplacerGranite");
    explicit PositionIDsReplacerGranite(const Output<Node>& unsq_position_ids);
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Replace implicit text-only Qwen3.5 M-RoPE positions with explicit PagedAttention positions.
 *
 * The text-only export of Qwen3.5 generates positions from the past cache length and current sequence length.
 * After SDPAToPA moves tokens to the batch dimension, the sequence length becomes one, so that
 * implicit branch cannot represent prefill or packed requests. Rewire the existing Broadcast's
 * data input to explicit positions, preserving its target input and the downstream M-RoPE graph.
 *
 * We change from this:                         to this:
 *
 *  ┌──────────────────────┐                   ┌──────────────────────┐
 *  │ Range(P, P + S, 1)   │                   │ unsq_position_ids    │
 *  │ implicit positions   │                   │ [T, 1]               │
 *  └──────────┬───────────┘                   └──────────┬───────────┘
 *             │                                          │
 *  ┌──────────┴───────────┐                   ┌──────────┴───────────┐
 *  │ Reshape([1, 1, -1])  │                   │ NEW Unsqueeze(0)     │
 *  │                      │                   │ [1, T, 1]            │
 *  └──────────┬───────────┘                   └──────────┬───────────┘
 *             │ input 0                                  │ input 0
 *  ┌──────────┴───────────┐                   ┌──────────┴───────────┐
 *  │ Broadcast            ├── target input 1  │ Broadcast            ├── target input 1
 *  │ [3, T, 1]            │   [3, T, 1]       │ [3, T, 1]            │   [3, T, 1]
 *  └──────────┬───────────┘   (unchanged)     └──────────┬───────────┘   (unchanged)
 *             │                                          │
 *  ┌──────────┴───────────┐                   ┌──────────┴───────────┐
 *  │ Unsqueeze(2)         │                   │ Unsqueeze(2)         │
 *  │ [3, T, 1, 1]         │                   │ [3, T, 1, 1]         │
 *  └──────────┬───────────┘                   └──────────┬───────────┘
 *             │                                          │
 *  ┌──────────┴───────────┐                   ┌──────────┴───────────┐
 *  │ Convert              │                   │ Convert              │
 *  │ MatMul(inv_freq)     │                   │ MatMul(inv_freq)     │
 *  │ Transpose            │                   │ Transpose            │
 *  └──────────────────────┘                   └──────────────────────┘
 *
 * T is the total token count. Both sides show the PA layout: the original branch broadcasts
 * one implicit position across tokens; the replacement preserves each token's explicit position.
 * The target is Concat([3], Reshape(Gather(ShapeOf(hidden_states), 0, 0), [1]), [1]).
 * Since hidden_states has shape [T, features, 1], the target values are [3, T, 1].
 */
class ov::pass::PositionIDsReplacerQwen3_5 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PositionIDsReplacerQwen3_5");
    explicit PositionIDsReplacerQwen3_5(const Output<Node>& unsq_position_ids);
};
