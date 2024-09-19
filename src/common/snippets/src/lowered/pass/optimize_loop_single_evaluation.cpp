// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/optimize_loop_single_evaluation.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool OptimizeLoopSingleEvaluation::run(lowered::LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::OptimizeLoopSingleEvaluation")
    const auto& loop_manager = linear_ir.get_loop_manager();

    bool is_modified = false;
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = *expr_it;
        if (auto loop_end = ov::as_type_ptr<op::LoopEnd>(expr->get_node())) {
            const auto& loop_info = loop_manager->get_loop_info<ExpandedLoopInfo>(loop_end->get_id());
            if (loop_info->get_work_amount() == loop_info->get_increment()) {
                auto new_finalization_offsets = loop_end->get_finalization_offsets();
                const auto& ptr_increments = loop_end->get_ptr_increments();
                const auto work_amount_incr = static_cast<int64_t>(loop_end->get_increment());
                for (size_t i = 0; i < new_finalization_offsets.size(); i++) {
                    const auto ptr_shift = utils::dynamic_safe_mul(ptr_increments[i], work_amount_incr);
                    new_finalization_offsets[i] = utils::dynamic_safe_add(new_finalization_offsets[i], ptr_shift);
                }
                loop_end->set_finalization_offsets(new_finalization_offsets);
                loop_end->set_ptr_increments(std::vector<int64_t>(new_finalization_offsets.size(), 0));
                loop_end->set_evaluate_once(true);

                // Update the corresponding ExpandedLoopInfo
                loop_info->update_ptr_increments(loop_end->get_ptr_increments());
                loop_info->update_finalization_offsets(loop_end->get_finalization_offsets());
                loop_info->set_evaluate_once(true);

                is_modified = true;
            }
        }
    }
    return is_modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

