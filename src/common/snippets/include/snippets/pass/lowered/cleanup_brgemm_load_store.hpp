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
 * @interface CleanupBrgemmLoadStore
 * @brief Brgemm operations (as implemented in CPU backend) have implicit MemoryAcess semantics. Therefore Load/Stores
 *        around Brgemm will be removed in this pass. Pointer increments on the second input will be also be zeroed, since
 *        the whole second input is traversed on every iteration.
 * @ingroup snippets
 */
class CleanupBrgemmLoadStore : public LinearIRTransformation {
public:
    OPENVINO_RTTI("CleanupBrgemmLoadStore", "LinearIRTransformation")
    bool run(LoweredExprIR& linear_ir) override;
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
