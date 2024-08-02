// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace pass {

/**
 * @interface SetTPPLeadingDim
 * @brief TPP leading dimension depends on the operation it is connected to. If it's a Parameter or Result
 * we can compute LD based on shape, if it's a Buffer - we need to consider allocation shape. 
 * This transformation should be performed before InsertTailLoop because it may change graph connectivity for 1st and last iterations.
 * @ingroup snippets
 */
class SetTPPLeadingDim: public snippets::lowered::pass::RangedPass {
public:
    OPENVINO_RTTI("SetTPPLeadingDim", "Pass");
    SetTPPLeadingDim();
    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;
};

}  // namespace pass
}  // namespace tpp
}  // namespace intel_cpu
}  // namespace ov
