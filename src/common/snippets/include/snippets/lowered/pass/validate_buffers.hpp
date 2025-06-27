// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/rtti.hpp"
#include "pass.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface ValidateBuffers
 * @brief The pass validates buffer expression in Linear IR state
 * @ingroup snippets
 */
class ValidateBuffers : public RangedPass {
public:
    OPENVINO_RTTI("ValidateBuffers", "", RangedPass)
    ValidateBuffers() = default;
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;
};

}  // namespace ov::snippets::lowered::pass
