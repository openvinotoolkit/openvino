// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_specific_iterations.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

LinearIR::constExprIt InsertSpecificIterations::insert_copy_loop(LinearIR& linear_ir, const size_t loop_id, const LinearIR::constExprIt& insert_pos) {
    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto loop_bounds = loop_manager->get_loop_bounds(linear_ir, loop_id);
    ExpressionMap expression_map;
    const auto& loop_copy_range = LinearIR::deep_copy_range(loop_bounds.first, std::next(loop_bounds.second), expression_map);
    const auto new_loop_begin_pos = linear_ir.insert(insert_pos, loop_copy_range.begin(), loop_copy_range.end());
    const auto new_loop_end_pos = insert_pos;

    // Set the same pointers of Tensor Shape to the cloned loop
    for (LinearIR::constExprIt cloned_it = new_loop_begin_pos, original_it = loop_bounds.first; cloned_it != new_loop_end_pos; ++cloned_it, ++original_it) {
        const auto& cloned_expr = *cloned_it;
        const auto& original_expr = *original_it;
        for (size_t i = 0; i < original_expr->get_input_count(); ++i)
            cloned_expr->get_input_port_descriptor(i)->set_shape_ptr(original_expr->get_input_port_descriptor(i)->get_shape_ptr());
        for (size_t i = 0; i < original_expr->get_output_count(); ++i)
            cloned_expr->get_output_port_descriptor(i)->set_shape_ptr(original_expr->get_output_port_descriptor(i)->get_shape_ptr());
    }

    const auto original_loop_info = loop_manager->get_loop_info(loop_id);
    std::vector<LoopPort> new_entry_points, new_exit_points;
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
        const auto outer_loop_ids = LoopManager::get_outer_expr_loops(expr, loop_id);
        for (size_t i = 0; i < expr->get_input_count(); ++i)
            loop_manager->update_loops_port(outer_loop_ids, expr->get_input_port(i), {expr->get_input_port(i), new_expr->get_input_port(i)}, true);
        for (size_t i = 0; i < expr->get_output_count(); ++i)
            loop_manager->update_loops_port(outer_loop_ids, expr->get_output_port(i), {expr->get_output_port(i), new_expr->get_output_port(i)}, false);
    }

    const auto new_id = loop_manager->replace_with_new_loop(linear_ir, new_loop_begin_pos, new_loop_end_pos,
                                                            original_loop_info->get_work_amount(), original_loop_info->get_increment(),
                                                            new_entry_points, new_exit_points, loop_id);
    const auto loop_end = ov::as_type_ptr<op::LoopEndStatic>(std::prev(new_loop_end_pos)->get()->get_node());
    OPENVINO_ASSERT(loop_end, "Cloned Loop does not contain LoopEnd op at the expected place.");
    loop_end->set_id(new_id);
    return new_loop_begin_pos;
}

bool InsertSpecificIterations::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertSpecificIterations")
    const auto& loop_manager = linear_ir.get_loop_manager();

    bool modified = false;
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = *expr_it;
        const auto node = expr->get_node();
        const auto loop_end = ov::as_type_ptr<op::LoopEndStatic>(node);
        if (!loop_end)
            continue;

        const auto& loop_info = loop_manager->get_loop_info(loop_end->get_id());
        const auto work_amount = loop_info->get_work_amount();
        const auto increment = loop_info->get_increment();
        const auto& handlers = loop_info->get_handlers();

        const auto main_loop_begin_it = linear_ir.find(linear_ir.get_expr_by_node(loop_end->get_loop_begin()));
        const auto main_loop_end_it = linear_ir.find_after(main_loop_begin_it, linear_ir.get_expr_by_node(loop_end));
        // Note: handlers must be run on the range started with the first operation in the loop body.
        const auto main_first_body_op_it = std::next(main_loop_begin_it);

        auto update_loop_params = [&loop_manager](const std::shared_ptr<op::LoopEndStatic>& loop_end_copy,
                                                  size_t new_work_amount,
                                                  size_t new_increment,
                                                  bool zero_finalization_offsets) {
            loop_end_copy->set_work_amount(new_work_amount);
            loop_end_copy->set_increment(new_increment);

            const auto& loop_info_copy = loop_manager->get_loop_info(loop_end_copy->get_id());
            loop_info_copy->set_work_amount(new_work_amount);
            loop_info_copy->set_increment(new_increment);

            if (zero_finalization_offsets)
                loop_end_copy->set_finalization_offsets(std::vector<int64_t>(loop_end_copy->get_finalization_offsets().size(), 0));
        };

        auto copy_and_run_specific_handlers = [&](const PassPipeline& handlers) {
            const auto new_loop_begin_pos = insert_copy_loop(linear_ir, loop_end->get_id(), main_loop_begin_it);
            const auto new_loop_begin = ov::as_type_ptr<op::LoopBeginStatic>(new_loop_begin_pos->get()->get_node());
            OPENVINO_ASSERT(new_loop_begin, "Cloned Loop does not contain LoopBegin op at the expected place.");
            const auto new_loop_end = ov::as_type_ptr<op::LoopEndStatic>(new_loop_begin->get_loop_end());
            OPENVINO_ASSERT(new_loop_end, "Cloned Loop does not contain LoopEnd op at the expected place.");
            const auto new_loop_end_pos = linear_ir.find_after(new_loop_begin_pos, linear_ir.get_expr_by_node(new_loop_end));

            // Note: handlers must be run on the range started with the first operation in the loop body.
            handlers.run(linear_ir, std::next(new_loop_begin_pos), new_loop_end_pos);
            return new_loop_end;
        };

        const bool specific_first_iteration = !handlers.get_first_iter_handlers().empty();
        if (work_amount == increment) {
            handlers.get_first_iter_handlers().run(linear_ir, main_first_body_op_it, main_loop_end_it);
        } else {
            if (specific_first_iteration) {
                const auto loop_end_copy = copy_and_run_specific_handlers(handlers.get_first_iter_handlers());
                update_loop_params(loop_end_copy, increment, increment, true);
            }

            const auto tail_size = work_amount % increment;
            if (tail_size != 0) {
                if (!specific_first_iteration || work_amount > 2 * increment) {
                    const auto loop_end_copy = copy_and_run_specific_handlers(handlers.get_main_iter_handlers());
                    const auto reduce_value = specific_first_iteration ? tail_size + increment : tail_size;
                    const auto new_work_amount = work_amount - reduce_value;
                    update_loop_params(loop_end_copy, new_work_amount, increment, true);
                }
                handlers.get_last_iter_handlers().run(linear_ir, main_first_body_op_it, main_loop_end_it);
                update_loop_params(loop_end, tail_size, tail_size, false);
            } else if (specific_first_iteration) {
                handlers.get_main_iter_handlers().run(linear_ir, main_first_body_op_it, main_loop_end_it);
                update_loop_params(loop_end, work_amount - increment, increment, false);
            }
        }
        modified = true;
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

