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
 * @interface AssignRegisters
 * @brief Assigns in/out abstract registers indexes to every operation.
 * Note that changing of the IR is likely to invalidate register assignment.
 * @ingroup snippets
 */
class AssignRegisters : public LinearIRTransformation {
public:
    OPENVINO_RTTI("AssignRegisters", "LinearIRTransformation")
    bool run(LoweredExprIR& linear_ir) override;
    static constexpr size_t reg_count = 16lu;
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
