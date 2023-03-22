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
 * @interface MoveResultOutOfLoop
 * @brief After passes with Loop work results would be inside Loop. The pass extract them from Loop and insert after.
 * @ingroup snippets
 */
class MoveResultOutOfLoop : public LinearIRTransformation {
public:
    OPENVINO_RTTI("MoveResultOutOfLoop", "LinearIRTransformation")
    MoveResultOutOfLoop() = default;
    bool run(LoweredExprIR& linear_ir) override;
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
