// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface ValidateShapes
 * @brief The pass checks that there are no dynamic shapes in the IR
 * @ingroup snippets
 */
class ValidateShapes : public RangedPass {
public:
    OPENVINO_RTTI("ValidateShapes", "", RangedPass)
    ValidateShapes() = default;
    bool run(lowered::LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
