// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/rtti.hpp"
#include "pass.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface ValidateShapes
 * @brief The pass checks that there are no dynamic shapes in the IR
 * @ingroup snippets
 */
class ValidateShapes : public RangedPass {
public:
    OPENVINO_RTTI("ValidateShapes", "", RangedPass)
    ValidateShapes() = default;
    bool run(lowered::LinearIR& linear_ir,
             lowered::LinearIR::constExprIt begin,
             lowered::LinearIR::constExprIt end) override;
};

}  // namespace ov::snippets::lowered::pass
