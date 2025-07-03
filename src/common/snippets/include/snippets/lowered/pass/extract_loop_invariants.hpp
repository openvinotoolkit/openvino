// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/rtti.hpp"
#include "pass.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface ExtractLoopInvariants
 * @brief Extracts expressions that produce identical result on every loop iteration outside of the loop's body.
 *        This extraction is to remove repeated computation, not cover constant subgraph extraction.
 * @ingroup snippets
 */
class ExtractLoopInvariants : public RangedPass {
public:
    OPENVINO_RTTI("ExtractLoopInvariants", "", RangedPass);
    ExtractLoopInvariants() = default;
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;
};

}  // namespace ov::snippets::lowered::pass
