// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief L2SharedInputMatMulFusion transformation identifies multiple MatMul operations
 * sharing a common L2-normalized input and fuses them into a single MatMul followed by a VariadicSplit.
 *
 * Transformation looks for the following pattern:
 *
 *        input
 *          |
 *        Power(x, 2)
 *          |
 *      ReduceSum(axis)
 *          |
 *         Sqrt
 *          |
 *       Divide(x, sqrt)
 *          |
 *       Multiply(x, scale)  ← shared normalized input
 *        /     |     \
 *     MatMul  MatMul  MatMul  ← each with quantized or constant weights
 *
 * And rewrites into:
 *
 *        input
 *          |
 *       L2Normalize (Power → ReduceSum → Sqrt → Divide → Mul)
 *          |
 *       MatMul (with packed weights)
 *          |
 *       VariadicSplit (splits packed output back into separate Q, K, V, ...)
 *
 *
 */
class TRANSFORMATIONS_API PackQKVProj;

}  // namespace pass
}  // namespace ov

class ov::pass::PackQKVProj : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("L2SharedInputMatMulFusion");
    PackQKVProj();
};
