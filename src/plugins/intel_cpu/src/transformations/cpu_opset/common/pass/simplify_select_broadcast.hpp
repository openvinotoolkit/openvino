// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu::pass {

/**
 * @ingroup ov_transformation_cpu_api
 * @brief SimplifySelectBroadcast replaces Broadcast(scalar_const, dynamic_shape) with
 * the scalar_const directly when it appears as an input to a Select operation.
 *
 * This transformation handles a pattern found in hybrid SSM+attention models (e.g., Qwen3.5/
 * qwen3_5_text architecture) where the aten::where/Select operation receives a Broadcast of
 * a constant scalar whose shape is derived from the SSM layer output. This shape may be
 * incompatible with the Select condition tensor at runtime (e.g., during prefill when the
 * SSM sequence length differs from the full-attention KV cache length), causing a shape
 * inference failure in the EltwiseShapeInfer:
 *   "Eltwise shape infer input shapes dim index: 2 mismatch"
 *
 * Pattern:
 *   Select(condition, on_true, Broadcast(scalar_const, shape_from_somewhere))
 *   OR
 *   Select(condition, Broadcast(scalar_const, shape_from_somewhere), on_false)
 *
 * Replacement:
 *   Select(condition, on_true, scalar_const)
 *   OR
 *   Select(condition, scalar_const, on_false)
 *
 * Semantic correctness: Broadcast(scalar_const, shape) fills a tensor with the constant value.
 * Since Select with NUMPY broadcast already broadcasts scalar inputs to the output shape,
 * replacing Broadcast(scalar_const, shape) with scalar_const produces identical values. The
 * output shape is then determined by the broadcast of {condition, on_true/on_false (scalar)},
 * which avoids the shape incompatibility.
 *
 * Preconditions:
 *  1. The Select uses NUMPY auto-broadcast.
 *  2. One of the Select data inputs (on_true or on_false, i.e., input index 1 or 2) is a
 *     Broadcast node whose first input (the data being broadcast) is a scalar constant.
 *  3. The Broadcast is consumed only by the Select (consumers_count == 1).
 */
class SimplifySelectBroadcast : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SimplifySelectBroadcast");
    SimplifySelectBroadcast();
};

}  // namespace ov::intel_cpu::pass
