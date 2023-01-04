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
 * @interface PropagateLayout
 * @brief Propagate layout from Parameter child to parameter and from Result Parent to Result. This is needed to calculate
 * proper data pointer offsets in the Kernel;
 * @ingroup snippets
 */
class PropagateLayout : public LinearIRTransformation {
public:
    OPENVINO_RTTI("PropagateLayout", "LinearIRTransformation")
    bool run(LoweredExprIR& linear_ir) override;
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
