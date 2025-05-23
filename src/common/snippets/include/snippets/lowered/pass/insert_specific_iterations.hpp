// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/lowered/specific_loop_iter_types.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/loop.hpp"


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
    OPENVINO_RTTI("InsertSpecificIterations", "", RangedPass);
    InsertSpecificIterations() = default;
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

    /**
     * @brief Check if specific Loop iterations needed
     * @param unified_loop_info loop info of the original (unified) Loop
     * @param type type of the specific loop iterations
     * @param remaining_work_amount the work amount on the current moment (after applying of the previous loop decomposed parts)
     * @return True if needed otherwise - False
     */
    static bool is_decomposed_loop_needed(const UnifiedLoopInfoPtr& unified_loop_info, SpecificLoopIterType type, size_t remaining_work_amount);
    /**
     * @brief Calculate work amount of specific Loop iterations
     * @param unified_loop_info loop info of the original (unified) Loop
     * @param type type of the specific loop iterations
     * @param remaining_work_amount the work amount on the current moment (after applying of the previous loop decomposed parts)
     * @return work amount
     */
    static size_t get_decomposed_loop_work_amount(const UnifiedLoopInfoPtr& unified_loop_info, SpecificLoopIterType type, size_t remaining_work_amount);
    /**
     * @brief Calculate increment of specific Loop iterations
     * @param unified_loop_info loop info of the original (unified) Loop
     * @param type type of the specific loop iterations
     * @param remaining_work_amount the work amount on the current moment (after applying of the previous loop decomposed parts)
     * @return increment
     */
    static size_t get_decomposed_loop_increment(const UnifiedLoopInfoPtr& unified_loop_info, SpecificLoopIterType type, size_t remaining_work_amount);

private:
    /**
     * @brief Decomposes the original Loop to the several specific iterations
     * @param linear_ir target Linear IR
     * @param begin iterator of LoopBegin
     * @param end iterator of LoopEnd
     * @param loop_end the target LoopEnd
     * @return True if the Loop has been successfully decomposed, otherwise returns False.
     */
    static bool decompose(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end, const std::shared_ptr<op::LoopEnd>& loop_end);
    /**
     * @brief Make a copy of Loop with ID `loop_id` and insert to LinearIR before `insert_pos`
     * @param linear_ir target Linear IR
     * @param loop_id the target loop ID
     * @param insert_pos insertion position iterator
     * @param new_entry_ports reference of vector with Loop input ports that will be updated after insertion
     * @param new_exit_ports reference of vector with Loop output ports that will be updated after insertion
     * @return LoopBounds: iterators of new LoopBegin and LoopEnd
     */
    static LoopManager::LoopBounds insert_copy_loop(LinearIR& linear_ir, const size_t loop_id, const LinearIR::constExprIt& insert_pos,
                                                    std::vector<LoopPort>& new_entry_ports, std::vector<LoopPort>& new_exit_ports);
    /**
     * @brief Initializes decomposed loop: update ptr arithmetic, work_amout, increment, ID
     * @param linear_ir target Linear IR
     * @param begin iterator of LoopBegin
     * @param end iterator of LoopEnd
     * @param decomposed_loop_info loop info of the corresponding decomposed loop
     * @param unified_loop_id ID of the unified loop
     * @param decomposed_loop_end LoopEnd of the decomposed loop
     */
    static void init_decomposed_loop(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end,
                                     const ExpandedLoopInfoPtr& decomposed_loop_info, size_t unified_loop_id,
                                     const std::shared_ptr<op::LoopEnd>& decomposed_loop_end);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
