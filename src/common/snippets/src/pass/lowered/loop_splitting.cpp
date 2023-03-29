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
    size_t prev_loop_id = SIZE_MAX;

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto expr = *expr_it;
        const auto& node = expr->get_node();
        if (ov::is_type<opset1::Parameter>(node) ||
            ov::is_type<opset1::Constant>(node) ||
            ov::is_type<opset1::Result>(node))
            continue;

        // Note: we currently consider only the outermost loops for splitting
        // Splitting could also be done in a more general case, but the splitted loop and its parent must always
        // be in the same set of outer loops. Otherwise they won't be fused.
        const auto loop_id = expr->get_loop_ids().front();
        if (prev_loop_id != loop_id) {
            auto loop_info = loop_manager->get_loop_info(loop_id);
            const auto& entry_points = loop_info->entry_exprs;
            for (const auto& entry_point : entry_points) {
                const auto input_td = entry_point.expr->get_inputs()[entry_point.port];
                const auto parent_expr_output = linear_ir.get_expr_by_output(input_td);
                const auto parent_expr = parent_expr_output.expr;
                const auto parent = parent_expr->get_node();
                if (ov::is_type<opset1::Constant>(parent) ||
                    ov::is_type<opset1::Parameter>(parent)) {
                    continue;
                }
                const auto loop_id_target = parent_expr->get_loop_ids().front();
                if (loop_id_target == LoweredExpr::LOOP_NULL_ID)
                    continue;
                const auto& loop_info_target = loop_manager->get_loop_info(loop_id_target);
                if (must_be_split(loop_info, loop_info_target)) {
                    const auto work_amount_outer = loop_info->work_amount;
                    const auto increment_outer = loop_info_target->increment;
                    loop_info->work_amount = increment_outer;
                    const auto split_loop =  std::make_shared<LoweredLoopInfo>(work_amount_outer,
                                                                                   increment_outer,
                                                                                   loop_info->entry_exprs,
                                                                                   loop_info->exit_exprs);
                    const auto split_loop_id = loop_manager->add_loop_info(split_loop);

                    LoweredExprIR::constExprIt target_loop_begin_pos, target_loop_end_pos;
                    LoweredLoopManager::get_loop_bounds(linear_ir, loop_info_target->entry_exprs, loop_info_target->exit_exprs,
                                                        target_loop_begin_pos, target_loop_end_pos, loop_id_target);
                    for (auto it = target_loop_begin_pos; it != target_loop_end_pos; it++) {
                        auto iexpr = it->get();
                        // Note: There could be exprs inside loop bounds that don't belong to the loop
                        if (iexpr->get_loop_ids().front() == loop_id) {
                            const auto& original_loop_ids = iexpr->get_loop_ids();
                            std::vector<size_t> loop_ids {split_loop_id};
                            loop_ids.insert(loop_ids.end(), original_loop_ids.begin(), original_loop_ids.end());
                            iexpr->set_loop_ids(loop_ids);
                        }
                    }
                    break;
                }
            }
            prev_loop_id = expr->get_loop_ids().front();
        }
    }

    return true;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
