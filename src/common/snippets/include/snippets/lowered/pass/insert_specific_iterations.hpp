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
 * @interface InsertSpecificIterations
 * @brief Inserts separate loop bodies for first/last iterations if needed.
 * Also calls previously registered SpecificIterationHandlers for the inserted bodies and the main body.
 * @ingroup snippets
 */
class InsertSpecificIterations : public RangedPass {
public:
    OPENVINO_RTTI("InsertSpecificIterations", "RangedPass")
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

    /**
     * @brief Makes a copy of a loop body with id 'loop_id' and inserts it to the LinearIR before the 'insert_pos' position
     * @param linear_ir LinearIR which should be modified
     * @param loop_id id of the loop which should be copied
     * @param insert_pos position before which the loop body copy should be inserted
     * @return iterator which points on the LoopBegin copy
     */
    static LinearIR::constExprIt insert_copy_loop(LinearIR& linear_ir,
                                                  const size_t loop_id,
                                                  const LinearIR::constExprIt& insert_pos);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
