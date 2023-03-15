// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinkingUnaryForward;
class TRANSFORMATIONS_API TransposeSinkingUnaryBackward;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeSinkingUnaryForward transformation sinks Transpose through UnaryElementwiseArithmetic, Clamp, Elu,
 * SoftPlus, LogicalNot, Convert, IsInf, IsNaN, IsFinite operations in the forward direction.
 */
class ov::pass::TransposeSinkingUnaryForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeSinkingUnaryForward", "0");
    TransposeSinkingUnaryForward();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeSinkingUnaryBackward transformation sinks Transpose through UnaryElementwiseArithmetic, Clamp, Elu,
 * SoftPlus, LogicalNot, Convert, IsInf, IsNaN, IsFinite in the backward direction.
 */
class ov::pass::TransposeSinkingUnaryBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeSinkingUnaryBackwardMultiConsumers", "0");
    TransposeSinkingUnaryBackward();
};
