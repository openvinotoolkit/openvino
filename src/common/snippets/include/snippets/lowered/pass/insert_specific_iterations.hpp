// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "openvino/core/rtti.hpp"
#include "pass.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/loop_port.hpp"
#include "snippets/lowered/specific_loop_iter_types.hpp"
#include "snippets/op/loop.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @ interface InsertSpecificIterations
 * @brief Inserts separate loop bodies for first/last iterations if needed.
 * Also calls previously registered SpecificIterationHandlers for the inserted bodies and the main body.
 * @ingroup snippets
 */
class InsertSpecificIterations : public RangedPass {
public:
    OPENVINO_RTTI("InsertSpecificIterations", "", RangedPass);
    InsertSpecificIterations() = default;
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

    /**
     * @brief Check if specific Loop iterations needed
     * @param unified_loop_info loop info of the original (unified) Loop
     * @param type type of the specific loop iterations
     * @param remaining_work_amount the work amount on the current moment (after applying of the previous loop
     * decomposed parts)
     * @return True if needed otherwise - False
     */
    static bool is_decomposed_loop_needed(const UnifiedLoopInfoPtr& unified_loop_info,
                                          SpecificLoopIterType type,
                                          size_t remaining_work_amount);
    /**
     * @brief Calculate work amount of specific Loop iterations
     * @param unified_loop_info loop info of the original (unified) Loop
     * @param type type of the specific loop iterations
     * @param remaining_work_amount the work amount on the current moment (after applying of the previous loop
     * decomposed parts)
     * @return work amount
     */
    static size_t get_decomposed_loop_work_amount(const UnifiedLoopInfoPtr& unified_loop_info,
                                                  SpecificLoopIterType type,
                                                  size_t remaining_work_amount);
    /**
     * @brief Calculate increment of specific Loop iterations
     * @param unified_loop_info loop info of the original (unified) Loop
     * @param type type of the specific loop iterations
     * @param remaining_work_amount the work amount on the current moment (after applying of the previous loop
     * decomposed parts)
     * @return increment
     */
    static size_t get_decomposed_loop_increment(const UnifiedLoopInfoPtr& unified_loop_info,
                                                SpecificLoopIterType type,
                                                size_t remaining_work_amount);

private:
    /**
     * @brief Decomposes the original Loop to the several specific iterations
     * @param linear_ir target Linear IR
     * @param begin iterator of LoopBegin
     * @param end iterator of LoopEnd
     * @param loop_end the target LoopEnd
     * @return True if the Loop has been successfully decomposed, otherwise returns False.
     */
    static bool decompose(LinearIR& linear_ir,
                          LinearIR::constExprIt begin,
                          LinearIR::constExprIt end,
                          const std::shared_ptr<op::LoopEnd>& loop_end);
    /**
     * @brief Make a copy of Loop with ID `loop_id` and insert to LinearIR before `insert_pos`
     * @param linear_ir target Linear IR
     * @param bounds loop bounds of current loop
     * @param insert_pos insertion position iterator
     * @param expression_map expression map to store pairs [original_expr, new_expr]
     * @return LoopBounds: iterators of new LoopBegin and LoopEnd
     */
    static LoopManager::LoopBounds insert_copy_loop(LinearIR& linear_ir,
                                                    const LoopManager::LoopBounds& bounds,
                                                    const LinearIR::constExprIt& insert_pos,
                                                    ExpressionMap& expression_map);
    /**
     * @brief Initializes decomposed loop: update ptr arithmetic, work_amout, increment, ID
     * @param linear_ir target Linear IR
     * @param decomposed_loop_bounds decomposed loop bounds
     * @param decomposed_loop_info loop info of the corresponding decomposed loop
     * @param loop_id_to_replace ID of the loop which should be replaced by the decomposed one
     * @param decomposed_loop_end LoopEnd of the decomposed loop
     * @param run_handlers flag to run handlers for the decomposed loop
     */
    static void init_decomposed_loop(LinearIR& linear_ir,
                                     const LoopManager::LoopBounds& decomposed_loop_bounds,
                                     const ExpandedLoopInfoPtr& decomposed_loop_info,
                                     size_t loop_id_to_replace,
                                     const std::shared_ptr<op::LoopEnd>& decomposed_loop_end,
                                     bool run_handlers);
};

}  // namespace ov::snippets::lowered::pass
