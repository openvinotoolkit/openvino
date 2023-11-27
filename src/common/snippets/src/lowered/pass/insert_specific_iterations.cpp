// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_specific_iterations.hpp"
#include "snippets/lowered/pass/iter_handler.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

LinearIR::container InsertSpecificIterations::copy_loop(const LinearIR& linear_ir, const size_t loop_id) {
    const auto& loop_manager = linear_ir.get_loop_manager();
    LinearIR::constExprIt loop_begin_pos, loop_end_pos;
    loop_manager->get_loop_bounds(linear_ir, loop_id, loop_begin_pos, loop_end_pos, true);
    ExressionMap expression_map;
    const auto& loop_copy_range = LinearIR::deep_copy_range(loop_begin_pos, std::next(loop_end_pos), expression_map);

    const auto original_loop_info = loop_manager->get_loop_info(loop_id);
    std::vector<LinearIR::LoopManager::LoopPort> new_entry_points, new_exit_points;
    // Clone loop ports from original loop info to new loop info
    for (const auto& entry : original_loop_info->get_entry_points())
        new_entry_points.push_back(*entry.clone_with_new_expr(expression_map[entry.expr_port->get_expr().get()]));
    for (const auto& exit : original_loop_info->get_exit_points())
        new_exit_points.push_back(*exit.clone_with_new_expr(expression_map[exit.expr_port->get_expr().get()]));

    for (const auto& elem : expression_map) {
        const auto expr = elem.first->shared_from_this();
        const auto& new_expr = elem.second;
        // Loop begin/end ops can't be loop ports
        if (ov::is_type<op::LoopBase>(expr->get_node()))
            continue;
        // Update loop info of all outer loops with new loop ports
        const auto outer_loop_ids = LinearIR::LoopManager::get_outer_expr_loops(expr, loop_id);
        for (size_t i = 0; i < expr->get_input_count(); ++i)
            loop_manager->update_loops_port(outer_loop_ids, expr->get_input_port(i), {expr->get_input_port(i), new_expr->get_input_port(i)}, true);
        for (size_t i = 0; i < expr->get_output_count(); ++i)
            loop_manager->update_loops_port(outer_loop_ids, expr->get_output_port(i), {expr->get_output_port(i), new_expr->get_output_port(i)}, false);
    }

    const auto new_loop_begin_pos = loop_copy_range.begin();
    const auto new_loop_end_pos = loop_copy_range.end();
    const auto new_id = loop_manager->replace_with_new_loop(linear_ir,
                                                            std::next(new_loop_begin_pos),
                                                            std::prev(new_loop_end_pos),
                                                            original_loop_info->get_work_amount(),
                                                            original_loop_info->get_increment(),
                                                            new_entry_points,
                                                            new_exit_points,
                                                            loop_id);
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(std::prev(new_loop_end_pos)->get()->get_node());
    OPENVINO_ASSERT(loop_end, "Cloned Loop does not contain LoopEnd op at the expected place.");
    loop_end->set_id(new_id);
    return loop_copy_range;
}

bool InsertSpecificIterations::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertSpecificIterations")
    const auto& loop_manager = linear_ir.get_loop_manager();

    bool modified = false;
    for (auto expr_it = linear_ir.cbegin(); expr_it != linear_ir.cend(); ++expr_it) {
        const auto& expr = *expr_it;
        const auto node = expr->get_node();
        const auto loop_end = ov::as_type_ptr<op::LoopEnd>(node);
        if (!loop_end)
            continue;

        std::vector<lowered::pass::SubgraphPassPipeline> pipelines_to_run;
        for (const auto& handlers : loop_manager->get_loop_info(loop_end->get_id())->handlers) {
            if (!handlers.empty())
                pipelines_to_run.emplace_back(handlers);
        }
        if (pipelines_to_run.empty())
            continue;

        const auto main_body_begin_it = linear_ir.find(linear_ir.get_expr_by_node(loop_end->get_loop_begin()));
        const auto main_body_end_it = linear_ir.find(linear_ir.get_expr_by_node(loop_end));
        auto copy_and_run_specific_handlers = [&](const SubgraphPassPipeline& handlers) {
            const auto& cloned_body = copy_loop(linear_ir, loop_end->get_id());
            linear_ir.insert(main_body_begin_it, cloned_body.begin(), cloned_body.end());
            handlers.run(linear_ir, cloned_body.begin(), std::prev(cloned_body.end()));
        };

        for (size_t i = 0; i < pipelines_to_run.size() - 1; ++i) {
            copy_and_run_specific_handlers(pipelines_to_run[i]);
        }
        // Last pipeline is run on original body to avoid unnecesarry copy
        pipelines_to_run.back().run(linear_ir, main_body_begin_it, main_body_end_it);
        modified = true;
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

