// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/rtti.hpp"
#include "pass.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface OptimizeLoopSingleEvaluation
 * @brief Does the following optimizations if the Loop body can be executed only once:
 *        - sets evaluate_once parameter to true
 *        - moves all ptr arithmetic to finalization offsets
 * @ingroup snippets
 */
class OptimizeLoopSingleEvaluation : public RangedPass {
public:
    OPENVINO_RTTI("OptimizeLoopSingleEvaluation", "", RangedPass);
    bool run(lowered::LinearIR& linear_ir,
             lowered::LinearIR::constExprIt begin,
             lowered::LinearIR::constExprIt end) override;
};

}  // namespace ov::snippets::lowered::pass
