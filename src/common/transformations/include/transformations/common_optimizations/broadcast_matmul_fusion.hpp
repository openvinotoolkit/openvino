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
 * @brief Removes a redundant Broadcast that expands one MatMul input's batch dimensions.
 *
 * Matches the Data -> Broadcast -> MatMul pattern, with the Broadcast on either MatMul
 * input and Data being an arbitrary input (not necessarily a Constant). MatMul broadcasts
 * the batch (leading) dimensions of its operands implicitly, so an explicit Broadcast that
 * only expands those dimensions is redundant. The Broadcast is detached and Data is
 * connected to the MatMul directly; the now-dangling Broadcast and its target-shape
 * subgraph are left unreferenced and removed by later clean-up.
 *
 * A common source is the rotary-embedding branch, where a constant (e.g. inv_freq) is
 * expanded to the batch dimension taken from another input's shape. Detaching the
 * Broadcast also lets that shape-providing input become removable when it has no other
 * consumers (for example attention_mask during PagedAttention conversion).
 *
 * Before:
 *
 *     Data          Other
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
 *     Data          Other
 *       │             │
 *       └──────┬──────┘
 *           ┌──┴───┐
 *           │MatMul│
 *           └──────┘
 *
 * The Broadcast is removed only when it does not change the MatMul result:
 *  - the matrix (last two) dimensions are left intact by the Broadcast;
 *  - for every expanded batch dimension, the other MatMul operand carries the same
 *    dimension, proven equal by static value or by shape symbol; an unlabeled dynamic
 *    dimension is never assumed compatible, since that could hide a runtime batch mismatch
 *    the Broadcast would have rejected.
 *
 * The Broadcast node itself is not removed and may keep other consumers; only the matched
 * MatMul input is rewired to Data directly.
 */
class ov::pass::BroadcastMatMulFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("BroadcastMatMulFusion");
    BroadcastMatMulFusion();
};
