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
    return current_work_amount == target_work_amount &&
           target_increment != current_increment &&
           current_increment != 1 &&
           target_increment % current_increment == 0;
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
        if (loop_ids.empty() || loop_ids.front() == prev_loop_id || loop_ids.front() >= LoweredExpr::LOOP_NULL_ID)
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
            if (loop_ids_parent.empty() || loop_ids_parent.front() >= LoweredExpr::LOOP_NULL_ID)
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
                    const auto& iexpr = *it;
                    // Note: There could be exprs inside loop bounds that don't belong to the loop
                    if (iexpr->get_loop_ids().front() == loop_id) {
                        auto split_loop_ids = iexpr->get_loop_ids();
                        split_loop_ids.insert(split_loop_ids.begin(), split_loop_id);
                        iexpr->set_loop_ids(split_loop_ids);

                        // split input tensors for all expressions in the loop
                        std::vector<TensorDescriptorPtr> split_inputs;
                        auto original_inputs = iexpr->get_inputs();
                        for (size_t i = 0; i < original_inputs.size(); i++) {
                            auto td = original_inputs[i];
                            const auto& tensor = td->get_tensor();
                            const auto& layout = td->get_layout();
                            size_t dim_idx = tensor.size() - loop_ids.size();
                            OPENVINO_ASSERT(tensor[dim_idx] == work_amount_outer,
                                            "Can't split dimension: inconsistent work_amount and tensor shape");
                            std::vector<size_t> split_tensor(tensor);
                            split_tensor[dim_idx] = loop_info->work_amount;
                            split_tensor.insert(split_tensor.begin() + dim_idx, split_loop->work_amount / loop_info->work_amount);
                            //split_tensor.insert(split_tensor.begin() + dim_idx, 1);
                            std::vector<size_t> split_layout(layout);
                            for (auto &d : split_layout) {
                                if (d > split_layout[dim_idx])
                                    d++;
                            }
                            split_layout.insert(split_layout.begin() + dim_idx + 1, split_layout[dim_idx] + 1);
                            auto split_td = std::make_shared<TensorDescriptor>(split_tensor,
                                                                               td->get_subtensor(),
                                                                               split_layout);
                            linear_ir.replace_input(iexpr, i, split_td);
                            linear_ir.replace_output(linear_ir.get_expr_by_output(td), split_td);
                        }
                    }
                }
                // split output tensors for loop exit points
                for (const auto& exit_point : loop_info->exit_exprs) {
                    auto td = exit_point.expr->get_outputs()[exit_point.port];
                    const auto& tensor = td->get_tensor();
                    const auto& layout = td->get_layout();
                    size_t dim_idx = tensor.size() - loop_ids.size();
                    OPENVINO_ASSERT(tensor[dim_idx] == work_amount_outer,
                                    "Can't split dimension: inconsistent work_amount and tensor shape");
                    std::vector<size_t> split_tensor(tensor);
                    split_tensor[dim_idx] = loop_info->work_amount;
                    split_tensor.insert(split_tensor.begin() + dim_idx, split_loop->work_amount / loop_info->work_amount);
                    //split_tensor.insert(split_tensor.begin() + dim_idx, 1);
                    std::vector<size_t> split_layout(layout);
                    for (auto &d : split_layout) {
                        if (d > split_layout[dim_idx])
                            d++;
                    }
                    split_layout.insert(split_layout.begin() + dim_idx + 1, split_layout[dim_idx] + 1);
                    auto split_td = std::make_shared<TensorDescriptor>(split_tensor,
                                                                       td->get_subtensor(),
                                                                       split_layout);
                    linear_ir.replace_output(exit_point.expr, exit_point.port, split_td);
                    // Note: we have to copy set of exprs before replacing, since get_exprs_by_input returns
                    // reference to a set that will be modified in replace_input
                    auto to_replace = linear_ir.get_exprs_by_input(td);
                    for (const auto& consumer : to_replace)
                        linear_ir.replace_input(consumer.expr, consumer.port, split_td);
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
                loop_ids.push_back(LoweredExpr::LOOP_FAKE_ID);
//                loop_ids.insert(loop_ids.begin(), LoweredExpr::LOOP_FAKE_ID);
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
