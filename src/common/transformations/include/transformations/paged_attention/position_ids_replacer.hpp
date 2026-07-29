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
#include "openvino/pass/sdpa_to_paged_attention.hpp"
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
 * Gather(index=0, axis=0) and removes the Gather, reconnecting its consumers to the Gather's data input.
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
 * (Unsqueeze -> MatMul(inv_freq) -> Cos/Sin -> Multiply(scale) -> Broadcast -> Unsqueeze(axis=0)) and
 * rewrites the trailing Unsqueeze's axis from 0 to 1, moving the flattened-tokens axis to index 0 to match
 * the layout Q/K arrive in. This is independent of how the per-token positions were derived (works whether
 * or not EliminateDropBatch has already collapsed a batch-drop select feeding into the outer product).
 */
class ov::pass::RoPEUnsqueezeAxisReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEUnsqueezeAxisReplacer");
    RoPEUnsqueezeAxisReplacer();
};
