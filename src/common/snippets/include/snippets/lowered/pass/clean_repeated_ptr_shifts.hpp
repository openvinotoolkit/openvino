// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/rtti.hpp"
#include "pass.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface CleanRepeatedDataPointerShifts
 * @brief The pass `fuses` (reset) ptr increments and finalization offsets for ports of Loop
 *        with the same data expression (Buffer with the same ID, the same parent of Loads) to avoid double ptr shifts
 *        Note: Buffer always employ inplace logics by default. It means that if a loop has both
 *              an input and an output connected to Buffers, the corresponding register should nevertheless be
 *              incremented only once (because when the input reg is incremented, output incremented automatically).
 *              This condition should be removed when Buffers stop being inplace by default.
 * @ingroup snippets
 */
class CleanRepeatedDataPointerShifts : public RangedPass {
public:
    OPENVINO_RTTI("CleanRepeatedDataPointerShifts", "", RangedPass)
    CleanRepeatedDataPointerShifts() = default;

    bool run(lowered::LinearIR& linear_ir,
             lowered::LinearIR::constExprIt begin,
             lowered::LinearIR::constExprIt end) override;

private:
    static bool reuse_increments(const LoopManagerPtr& loop_manager, const ExpressionPtr& loop_end_expr);
};

}  // namespace ov::snippets::lowered::pass
