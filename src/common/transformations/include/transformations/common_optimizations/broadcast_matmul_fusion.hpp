// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API BroadcastMatMulFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Removes a redundant Broadcast that expands a Constant on a MatMul input.
 *
 * Matches the Constant -> Broadcast -> MatMul pattern, with the Broadcast on either
 * MatMul input. MatMul broadcasts the batch (leading) dimensions of its operands
 * implicitly, so an explicit Broadcast that only expands those dimensions of a
 * constant is redundant. The Broadcast is detached and the Constant is connected to
 * the MatMul directly; the now-dangling Broadcast and its target-shape subgraph are
 * left unreferenced and removed by later clean-up.
 *
 * A common source is the rotary-embedding branch, where a constant (e.g. inv_freq) is
 * expanded to the batch dimension taken from another input's shape. Detaching the
 * Broadcast also lets that shape-providing input become removable when it has no other
 * consumers (for example attention_mask during PagedAttention conversion).
 *
 * Before:
 *
 *   Constant        Other
 *       │             │
 *   ┌───┴─────┐       │
 *   │Broadcast│       │
 *   └───┬─────┘       │
 *       │             │
 *       └──────┬──────┘
 *           ┌──┴───┐
 *           │MatMul│
 *           └──────┘
 *
 * After:
 *
 *   Constant        Other
 *       │             │
 *       └──────┬──────┘
 *           ┌──┴───┐
 *           │MatMul│
 *           └──────┘
 *
 * The Broadcast is removed only when it does not change the MatMul result:
 *  - the Broadcast has a single consumer (the MatMul);
 *  - the matrix (last two) dimensions are left intact by the Broadcast;
 *  - for every expanded batch dimension, the other MatMul operand carries the same
 *    dimension. The other operand is considered to carry the dimension when it is
 *    provably equal (equal static value or equal shape symbol) or dynamic
 *    (runtime-compatible); a differing static value skips the match.
 */
class ov::pass::BroadcastMatMulFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("BroadcastMatMulFusion");
    BroadcastMatMulFusion();
};
