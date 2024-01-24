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

using LoopInfo = LinearIR::LoopManager::LoopInfo;

std::array<RuntimeConfig::LoopDescriptor::Type, 3> InsertSpecificIterations::m_loop_types = {
    RuntimeConfig::LoopDescriptor::Type::First,
    RuntimeConfig::LoopDescriptor::Type::Main,
    RuntimeConfig::LoopDescriptor::Type::Last
};

LinearIR::constExprIt InsertSpecificIterations::insert_copy_loop(LinearIR& linear_ir, const size_t loop_id, const LinearIR::constExprIt& insert_pos) {
    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto loop_bounds = loop_manager->get_loop_bounds(linear_ir, loop_id);
    ExpressionMap expression_map;
    const auto& loop_copy_range = LinearIR::deep_copy_range(loop_bounds.first, std::next(loop_bounds.second), expression_map);
    const auto new_loop_begin_pos = linear_ir.insert(insert_pos, loop_copy_range.begin(), loop_copy_range.end());
    const auto new_loop_end_pos = insert_pos;

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

    const auto new_id = loop_manager->replace_with_new_loop(linear_ir, new_loop_begin_pos, new_loop_end_pos,
                                                            original_loop_info->get_work_amount(), original_loop_info->get_increment(),
                                                            new_entry_points, new_exit_points, loop_id);
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(std::prev(new_loop_end_pos)->get()->get_node());
    OPENVINO_ASSERT(loop_end, "Cloned Loop does not contain LoopEnd op at the expected place.");
    loop_end->set_id(new_id);
    return new_loop_begin_pos;
}

PassPipeline InsertSpecificIterations::get_iter_specific_handlers_by_type(const LinearIR::LoopManager::LoopInfoPtr& loop_info,
                                                                          const RuntimeConfig::LoopDescriptor::Type& type) {
     switch (type) {
        case RuntimeConfig::LoopDescriptor::Type::First:
            return loop_info->get_handlers().get_first_iter_handlers();
        case RuntimeConfig::LoopDescriptor::Type::Main:
            return loop_info->get_handlers().get_main_iter_handlers();
        case RuntimeConfig::LoopDescriptor::Type::Last:
            return loop_info->get_handlers().get_last_iter_handlers();
        default:
            OPENVINO_THROW("Unknown LoopDescriptor type!");
    }
}

void InsertSpecificIterations::init_specific_loop(const std::shared_ptr<op::LoopEnd>& loop_end, const RuntimeConfig::LoopDescriptor& desc,
                                                  LinearIR& linear_ir, const PassPipeline& handlers,
                                                  LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    loop_end->set_desc_id(desc.id);
    loop_end->update(desc);
    OPENVINO_ASSERT(ov::is_type<op::LoopBegin>(begin->get()->get_node()), "Expected LoopBegin");
    OPENVINO_ASSERT(ov::is_type<op::LoopEnd>(end->get()->get_node()), "Expected LoopEnd");
    // Note: handlers must be run on the range started with the first operation in the loop body.
    handlers.run(linear_ir, std::next(begin), end);
}

bool InsertSpecificIterations::create_specific_loop(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end,
                                                    const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t original_loop_id,
                                                    const RuntimeConfig& runtime_config, const std::shared_ptr<op::LoopEnd>& loop_end,
                                                    const RuntimeConfig::LoopDescriptor::Type& type) {
    auto is_there_non_inserted_loops = [&](const RuntimeConfig::LoopDescriptor::Type& current_type) {
        const auto current_type_it = std::find(m_loop_types.cbegin(), m_loop_types.cend(), current_type);
        OPENVINO_ASSERT(current_type_it != m_loop_types.cend(), "Loop Type has not been found!");
        return std::any_of(std::next(current_type_it), m_loop_types.cend(),
                           [&](RuntimeConfig::LoopDescriptor::Type loop_type) { return runtime_config.contains(original_loop_id, loop_type); });
    };

    RuntimeConfig::LoopDescriptor loop_desc;
    if (runtime_config.get_loop_desc(original_loop_id, type, loop_desc)) {
        const auto handlers = get_iter_specific_handlers_by_type(loop_info, type);
        auto spec_loop_end = loop_end;
        auto spec_loop_begin_it = begin, spec_loop_end_it = end;
        // Need to copy body if there are other specific sup-loops
        // Otherwise we should update the current body
        if (is_there_non_inserted_loops(type)) {
            spec_loop_begin_it = insert_copy_loop(linear_ir, original_loop_id, begin);
            const auto new_loop_begin = ov::as_type_ptr<op::LoopBegin>(spec_loop_begin_it->get()->get_node());
            OPENVINO_ASSERT(new_loop_begin, "Cloned Loop does not contain LoopBegin op at the expected place.");
            spec_loop_end = new_loop_begin->get_loop_end();
            spec_loop_end_it = linear_ir.find_after(spec_loop_begin_it, linear_ir.get_expr_by_node(spec_loop_end));
        }
        init_specific_loop(spec_loop_end, loop_desc, linear_ir, handlers, spec_loop_begin_it, spec_loop_end_it);
        return true;
    }
    return false;
}

bool InsertSpecificIterations::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertSpecificIterations")
    bool modified = false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& runtime_config = linear_ir.get_lowered_config();

    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        if (const auto loop_end = ov::as_type_ptr<op::LoopEnd>(expr_it->get()->get_node())) {
            const auto begin_it = linear_ir.find_before(expr_it, linear_ir.get_expr_by_node(loop_end->get_loop_begin()));
            const auto end_it = expr_it;

            const auto loop_id = loop_end->get_id();
            const auto& loop_info = loop_manager->get_loop_info(loop_id);
            OPENVINO_ASSERT(runtime_config.contains(loop_id), "LoopDescriptors are missed for Loop with ID " + std::to_string(loop_id));

            bool created = false;
            for (const auto& type : m_loop_types)
                created = create_specific_loop(linear_ir, begin_it, end_it, loop_info, loop_id, runtime_config, loop_end, type) || created;
            OPENVINO_ASSERT(created, "The Loop has not been updated!");
            modified = true;
        }
    }

    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

