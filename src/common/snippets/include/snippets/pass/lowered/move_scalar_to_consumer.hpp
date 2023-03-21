// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_IR_transformation.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

/**
 * @interface MoveScalarToConsumer
 * @brief As a result of loop insertion or fusion, Scalar operations might end up outside of the loop where their
 *        consumer is located. This transformation moves every scalar right before its consumer. This is needed to guarantee
 *        computation validity and also to optimize register allocation.
 *        Details:
 *             If ScalarEmitters are called outside the Loop, and only the first Loop iteration would yield correct data
 *             (assuming the vector reg assigned to scalar will get corrupted inside the loop body).
 *             To avoid such cases, we move Constants to the places in Linear IR before right Consumer to execute Scalar on each Loop iteration.
 * @ingroup snippets
 */
class MoveScalarToConsumer : public LinearIRTransformation {
public:
    OPENVINO_RTTI("MoveScalarsToConsumer", "LinearIRTransformation")
    MoveScalarToConsumer() = default;
    bool run(LoweredExprIR& linear_ir) override;
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
