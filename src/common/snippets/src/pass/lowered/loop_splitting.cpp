// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/loop_splitting.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

using LoweredLoopManager = LoweredExprIR::LoweredLoopManager;
using LoweredLoopInfo = LoweredLoopManager::LoweredLoopInfo;
using LoweredLoopInfoPtr = LoweredLoopManager::LoweredLoopInfoPtr;

LoopSplitting::LoopSplitting() : LinearIRTransformation() {}

bool LoopSplitting::must_be_split(const LoweredExprIR::LoweredLoopManager::LoweredLoopInfoPtr& current,
                                     const LoweredExprIR::LoweredLoopManager::LoweredLoopInfoPtr& target) {
    auto current_work_amount = current->work_amount;
    auto current_increment = current->increment;
    auto target_work_amount = target->work_amount;
    auto target_increment = target->increment;
    // Note: work amounts must be the same, since work amount defines total pointer shift
    //       target increment must ve divisible by current increment, so the splitting wouldn't affect data access
    return current_work_amount == target_work_amount && target_increment % current_increment == 0;
}

bool LoopSplitting::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::LoopFusion")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    size_t prev_loop_id = LoweredExpr::LOOP_NULL_ID;
    bool loop_was_splitted = false;
    for (const auto& expr : linear_ir) {
        const auto& loop_ids = expr->get_loop_ids();
        if (loop_ids.empty() || loop_ids.front() == prev_loop_id || loop_ids.front() == LoweredExpr::LOOP_NULL_ID)
            continue;

        // Note: we currently consider only the outermost loops for splitting
        // Splitting could also be done in a more general case, but the splitted loop and its parent must always
        // be in the same set of outer loops. Otherwise they won't be fused.
        const auto loop_id = loop_ids.front();
        auto loop_info = loop_manager->get_loop_info(loop_id);
        for (const auto& entry_point : loop_info->entry_exprs) {
            const auto input_td = entry_point.expr->get_inputs()[entry_point.port];
            const auto expr_parent = linear_ir.get_expr_by_output(input_td).expr;
            const auto& loop_ids_parent = expr_parent->get_loop_ids();
            if (loop_ids_parent.empty() || loop_ids_parent.front() == LoweredExpr::LOOP_NULL_ID)
                continue;

            const auto loop_id_target = loop_ids_parent.front();
            const auto& loop_info_target = loop_manager->get_loop_info(loop_id_target);
            if (must_be_split(loop_info, loop_info_target)) {
                loop_was_splitted = true;
                const auto work_amount_outer = loop_info->work_amount;
                const auto increment_outer = loop_info_target->increment;
                loop_info->work_amount = increment_outer;
                const auto split_loop =  std::make_shared<LoweredLoopInfo>(work_amount_outer,
                                                                               increment_outer,
                                                                               loop_info->entry_exprs,
                                                                               loop_info->exit_exprs);
                const auto split_loop_id = loop_manager->add_loop_info(split_loop);

                LoweredExprIR::constExprIt loop_begin_pos, loop_end_pos;
                LoweredLoopManager::get_loop_bounds(linear_ir, loop_info->entry_exprs, loop_info->exit_exprs,
                                                    loop_begin_pos, loop_end_pos, loop_id_target);
                for (auto it = loop_begin_pos; it != loop_end_pos; it++) {
                    auto iexpr = it->get();
                    // Note: There could be exprs inside loop bounds that don't belong to the loop
                    if (iexpr->get_loop_ids().front() == loop_id) {
                        auto split_loop_ids = iexpr->get_loop_ids();
                        split_loop_ids.insert(split_loop_ids.begin(), split_loop_id);
                        iexpr->set_loop_ids(split_loop_ids);
                    }
                }
                break;
            }
        }
        prev_loop_id = expr->get_loop_ids().front();
    }
    if (loop_was_splitted) {
        for (const auto& expr : linear_ir) {
            auto loop_ids = expr->get_loop_ids();
            if (!loop_ids.empty() && loop_ids.size() < 3) {
                loop_ids.insert(loop_ids.begin(), LoweredExpr::LOOP_NULL_ID);
                expr->set_loop_ids(loop_ids);
            }
        }
    }

    return loop_was_splitted;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
