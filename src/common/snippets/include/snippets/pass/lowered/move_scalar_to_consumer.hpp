// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_IR_transformation.hpp"
#include "snippets/tensor_descriptor.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

/**
 * @interface MoveScalarToConsumer
 * @brief As a result of loop insertion or fusion, Scalar operations might end up outside of the loop where their
 *        consumer is located. This transformation moves every scalar right before its consumer. This is needed to guarantee
 *        computation validity and also to optimize register allocation.
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
