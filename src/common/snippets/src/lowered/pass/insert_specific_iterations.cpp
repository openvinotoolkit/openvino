// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_specific_iterations.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/linear_ir_builder.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_port.hpp"
#include "snippets/lowered/port_connector.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/lowered/specific_loop_iter_types.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::lowered::pass {

namespace {
std::vector<LoopPort> clone_ports(const ExpressionMap& expression_map, const std::vector<LoopPort>& cur_ports) {
    std::vector<LoopPort> new_ports(cur_ports.size());
    for (size_t i = 0; i < cur_ports.size(); ++i) {
        const auto& port = cur_ports[i];
        const auto& original_expr = port.get_expr_port()->get_expr().get();
        OPENVINO_ASSERT(expression_map.count(original_expr), "Cannot find cloned expression for: ", original_expr);
        new_ports[i] = *port.clone_with_new_expr(expression_map.at(original_expr));
    }
    return new_ports;
}

void connect_cloned_body_with_expr_outside_loop(const LoopManager::LoopBounds& cur_bounds,
                                                const LoopManager::LoopBounds& res_bounds,
                                                LinearIR& linear_ir) {
    const auto& [cur_begin, cur_end] = cur_bounds;
    const auto& [res_begin, res_end] = res_bounds;
    for (auto result_it = res_begin, original_it = cur_begin; result_it != res_end; ++result_it, ++original_it) {
        const auto& result_expr = *result_it;
        const auto& original_expr = *original_it;
        for (size_t i = 0; i < original_expr->get_output_count(); i++) {
            const auto& consumers = original_expr->get_output_port_connector(i)->get_consumers();
            for (const auto& consumer : consumers) {
                const auto consumer_expr = consumer.get_expr();
                // these expressions should be connected from all expanded loop for correct register assignment.
                if (utils::need_full_connectors(consumer_expr) &&
                    std::find(cur_begin, cur_end, consumer_expr) == cur_end) {
                    std::vector<PortDescriptorPtr> new_descs = {
                        consumer_expr->get_input_port_descriptor(consumer.get_index())->clone()};
                    std::vector<PortConnectorPtr> new_inputs = {result_expr->get_output_port_connector(i)};
                    OutputVector new_op_inputs = {result_expr->get_node()->output(i)};
                    for (size_t j = 0; j < consumer_expr->get_input_count(); ++j) {
                        const auto& source = consumer_expr->get_input_port_connector(j)->get_source();
                        new_op_inputs.push_back(source.get_expr()->get_node()->output(source.get_index()));
                        new_descs.push_back(consumer_expr->get_input_port_descriptor(j)->clone());
                        new_inputs.push_back(consumer_expr->get_input_port_connector(j));
                    }
                    const auto new_consumer_op = consumer_expr->get_node()->clone_with_new_inputs(new_op_inputs);
                    linear_ir.replace_with_expr(
                        {consumer_expr},
                        consumer_expr->clone_with_new_inputs(new_consumer_op, new_inputs, new_descs));
                    break;
                }
            }
        }
    }
}
}  // namespace

bool InsertSpecificIterations::is_decomposed_loop_needed(const UnifiedLoopInfoPtr& unified_loop_info,
                                                         SpecificLoopIterType type,
                                                         size_t remaining_work_amount) {
    OPENVINO_ASSERT(unified_loop_info, "UnifiedLoopInfo is missed!");
    const auto increment = unified_loop_info->get_increment();
    OPENVINO_ASSERT(!utils::is_dynamic_value(increment) && increment > 0, "Incorrect increment: ", increment);
    const auto is_dynamic = utils::is_dynamic_value(remaining_work_amount);

    switch (type) {
    case (SpecificLoopIterType::FIRST_ITER):
        return !unified_loop_info->get_handlers().get_passes<SpecificLoopIterType::FIRST_ITER>().empty() &&
               (is_dynamic || remaining_work_amount >= increment);
    case (SpecificLoopIterType::MAIN_BODY):
        return is_dynamic || remaining_work_amount >= increment;
    case (SpecificLoopIterType::LAST_ITER):
        return (is_dynamic && increment > 1) || (!is_dynamic && remaining_work_amount > 0);
    default:
        OPENVINO_THROW("Unknown SpecificLoopIterType!");
    }
}
size_t InsertSpecificIterations::get_decomposed_loop_work_amount(const UnifiedLoopInfoPtr& unified_loop_info,
                                                                 SpecificLoopIterType type,
                                                                 size_t remaining_work_amount) {
    OPENVINO_ASSERT(unified_loop_info, "UnifiedLoopInfo is missed!");
    const auto increment = unified_loop_info->get_increment();
    const auto is_dynamic = utils::is_dynamic_value(remaining_work_amount);

    switch (type) {
    case (SpecificLoopIterType::FIRST_ITER):
        // We don't set always `increment` for first iterations since in dynamic `work_amount` can be less than
        // `increment`
        return is_dynamic ? remaining_work_amount : increment;
    case (SpecificLoopIterType::MAIN_BODY):
        return is_dynamic ? remaining_work_amount : (remaining_work_amount / increment) * increment;
    case (SpecificLoopIterType::LAST_ITER): {
        OPENVINO_ASSERT(is_dynamic || remaining_work_amount < unified_loop_info->get_increment(),
                        "Last iter work amount (",
                        remaining_work_amount,
                        ") must be less than the UnifiedLoopInfo's increment: ",
                        unified_loop_info->get_increment());
        return remaining_work_amount;
    }
    default:
        OPENVINO_THROW("Unknown SpecificLoopIterType!");
    }
}
size_t InsertSpecificIterations::get_decomposed_loop_increment(const UnifiedLoopInfoPtr& unified_loop_info,
                                                               SpecificLoopIterType type,
                                                               size_t remaining_work_amount) {
    OPENVINO_ASSERT(unified_loop_info, "UnifiedLoopInfo is missed!");
    const auto increment = unified_loop_info->get_increment();

    switch (type) {
    case (SpecificLoopIterType::FIRST_ITER):
    case (SpecificLoopIterType::MAIN_BODY):
        return increment;
    case (SpecificLoopIterType::LAST_ITER):
        return remaining_work_amount;
    default:
        OPENVINO_THROW("Unknown SpecificLoopIterType!");
    }
}

LoopManager::LoopBounds InsertSpecificIterations::insert_copy_loop(LinearIR& linear_ir,
                                                                   const LoopManager::LoopBounds& bounds,
                                                                   const LinearIR::constExprIt& insert_pos,
                                                                   ExpressionMap& expression_map) {
    const auto& [loop_begin_pos, loop_end_pos] = bounds;
    const auto& cloning_config = LinearIRBuilder::Config(false);
    const auto& loop_copy_range =
        LinearIRBuilder(cloning_config).clone_range(loop_begin_pos, std::next(loop_end_pos), expression_map);
    const auto new_loop_begin_pos = linear_ir.insert(insert_pos, loop_copy_range.begin(), loop_copy_range.end());
    const auto new_loop_end_pos = std::prev(insert_pos);
    return {new_loop_begin_pos, new_loop_end_pos};
}

void InsertSpecificIterations::init_decomposed_loop(LinearIR& linear_ir,
                                                    const LoopManager::LoopBounds& decomposed_loop_bounds,
                                                    const ExpandedLoopInfoPtr& decomposed_loop_info,
                                                    size_t loop_id_to_replace,
                                                    const std::shared_ptr<op::LoopEnd>& decomposed_loop_end,
                                                    bool run_handlers) {
    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto new_id = loop_manager->replace_with_new_loop(linear_ir,
                                                            decomposed_loop_bounds.first,
                                                            std::next(decomposed_loop_bounds.second),
                                                            decomposed_loop_info,
                                                            loop_id_to_replace);
    decomposed_loop_end->set_id(new_id);
    decomposed_loop_end->set_work_amount(decomposed_loop_info->get_work_amount());
    decomposed_loop_end->set_increment(decomposed_loop_info->get_increment());
    decomposed_loop_end->set_ptr_increments(decomposed_loop_info->get_ptr_increments());
    decomposed_loop_end->set_finalization_offsets(decomposed_loop_info->get_finalization_offsets());
    if (run_handlers) {
        const auto handlers = decomposed_loop_info->get_handler_passes();
        // Note: handlers must be run on the range started with the first operation in the loop body.
        handlers.run(linear_ir, std::next(decomposed_loop_bounds.first), decomposed_loop_bounds.second);
    }
}

bool InsertSpecificIterations::decompose(LinearIR& linear_ir,
                                         LinearIR::constExprIt begin,
                                         LinearIR::constExprIt end,
                                         const std::shared_ptr<op::LoopEnd>& loop_end) {
    const auto unified_loop_id = loop_end->get_id();
    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& unified_loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(unified_loop_id);

    auto remaining_work_amount = unified_loop_info->get_work_amount();
    const auto is_wa_dynamic = utils::is_dynamic_value(remaining_work_amount);

    static constexpr std::array<SpecificLoopIterType, 3> loop_iterations = {SpecificLoopIterType::FIRST_ITER,
                                                                            SpecificLoopIterType::MAIN_BODY,
                                                                            SpecificLoopIterType::LAST_ITER};

    auto decomposed = false;
    for (const auto& iter_type : loop_iterations) {
        if (is_decomposed_loop_needed(unified_loop_info, iter_type, remaining_work_amount)) {
            const auto work_amount =
                get_decomposed_loop_work_amount(unified_loop_info, iter_type, remaining_work_amount);
            const auto increment = get_decomposed_loop_increment(unified_loop_info, iter_type, remaining_work_amount);
            // Update remaining Loop work amount
            // Note: if work_amount is unknown and increment = 1, it means that a loop will iterate by whole work_amount
            if (!is_wa_dynamic || increment == 1 || iter_type == SpecificLoopIterType::LAST_ITER) {
                remaining_work_amount -= work_amount;
            }

            auto decomposed_loop_end = loop_end;
            LoopManager::LoopBounds decomposed_loop_bounds{begin, end};
            auto decomposed_loop_entry_ports = unified_loop_info->get_input_ports();
            auto decomposed_loop_exit_ports = unified_loop_info->get_output_ports();
            auto decomposed_ptr_increments = unified_loop_info->get_ptr_increments();
            auto decomposed_finalization_offsets = unified_loop_info->get_finalization_offsets();
            auto decomposed_data_sizes = unified_loop_info->get_data_sizes();
            // Need to copy body if there are other specific sup-loops
            // Otherwise we should update the current body
            if (remaining_work_amount > 0) {
                const auto cur_bounds = loop_manager->get_loop_bounds(linear_ir, unified_loop_id);
                ExpressionMap expression_map;
                decomposed_loop_bounds = insert_copy_loop(linear_ir, cur_bounds, begin, expression_map);

                // Add connections between output of cloned bodies and expressions outside loop from the current
                // LinearIR (these expressions are connections between Loops)
                connect_cloned_body_with_expr_outside_loop(cur_bounds, decomposed_loop_bounds, linear_ir);

                const auto original_loop_info = loop_manager->get_loop_info(unified_loop_id);
                decomposed_loop_entry_ports = clone_ports(expression_map, original_loop_info->get_input_ports());
                decomposed_loop_exit_ports = clone_ports(expression_map, original_loop_info->get_output_ports());

                decomposed_loop_end = ov::as_type_ptr<op::LoopEnd>(decomposed_loop_bounds.second->get()->get_node());
                OPENVINO_ASSERT(decomposed_loop_end, "Cloned Loop does not contain LoopEnd op at the expected place.");

                // Only latest loop iterations must have summarized finalization offsets!
                // Since we inserted copy before the latest iterations, these copies should have reseted offsets
                std::for_each(decomposed_finalization_offsets.begin(),
                              decomposed_finalization_offsets.end(),
                              [](int64_t& offset) {
                                  if (!utils::is_dynamic_value(offset)) {
                                      offset = 0;
                                  }
                              });

                // Note: all internal decomposed loops must be also cloned to avoid a situation
                // when 2 loops with the same ID exist in both specific iterations of the outer loop
                LoopInfoMap loop_info_map;
                for (auto it = std::next(decomposed_loop_bounds.first); it != decomposed_loop_bounds.second; ++it) {
                    auto internal_loop_end = ov::as_type_ptr<op::LoopEnd>(it->get()->get_node());
                    if (!internal_loop_end) {
                        continue;
                    }
                    const auto loop_begin = internal_loop_end->get_loop_begin();
                    auto begin_it = linear_ir.find(std::next(decomposed_loop_bounds.first),
                                                   it,
                                                   linear_ir.get_expr_by_node(loop_begin));
                    LoopManager::LoopBounds internal_loop_bounds{begin_it, it};
                    const auto internal_loop_id = internal_loop_end->get_id();
                    // Note: internal loops must be already decomposed to ExpandedLoops
                    const auto internal_loop_info = loop_manager->get_loop_info<ExpandedLoopInfo>(internal_loop_id);

                    if (auto inner_split_info = ov::as_type_ptr<InnerSplittedUnifiedLoopInfo>(
                            internal_loop_info->get_unified_loop_info())) {
                        const auto outer_loop_info = inner_split_info->get_outer_splitted_loop_info();

                        outer_loop_info->iterate_through_ports([&](const LoopPort& port) {
                            const auto expr = port.get_expr_port()->get_expr();
                            // Note: output loop info, whose ports are outside of the internal loop bounds,
                            // must be kept, not cloned
                            if (expr->get_exec_num() < cur_bounds.first->get()->get_exec_num() ||
                                expr->get_exec_num() > cur_bounds.second->get()->get_exec_num()) {
                                loop_info_map[outer_loop_info.get()] = outer_loop_info;
                            }
                        });
                    }
                    const auto cloned_loop_info = ov::as_type_ptr<ExpandedLoopInfo>(
                        internal_loop_info->clone_with_new_expr(expression_map, loop_info_map));
                    OPENVINO_ASSERT(cloned_loop_info,
                                    "Internal loop with ID ",
                                    internal_loop_id,
                                    " must have ExpandedLoopInfo type after cloning!");
                    init_decomposed_loop(linear_ir,
                                         internal_loop_bounds,
                                         cloned_loop_info,
                                         internal_loop_id,
                                         internal_loop_end,
                                         false);
                }
            }

            const auto decomposed_loop_info = std::make_shared<ExpandedLoopInfo>(work_amount,
                                                                                 increment,
                                                                                 decomposed_loop_entry_ports,
                                                                                 decomposed_loop_exit_ports,
                                                                                 decomposed_ptr_increments,
                                                                                 decomposed_finalization_offsets,
                                                                                 decomposed_data_sizes,
                                                                                 iter_type,
                                                                                 unified_loop_info);
            init_decomposed_loop(linear_ir,
                                 decomposed_loop_bounds,
                                 decomposed_loop_info,
                                 unified_loop_id,
                                 decomposed_loop_end,
                                 true);

            decomposed = true;
        }
    }

    return decomposed;
}

bool InsertSpecificIterations::run(LinearIR& linear_ir,
                                   lowered::LinearIR::constExprIt begin,
                                   lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertSpecificIterations")

    bool modified = false;
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        if (const auto loop_end = ov::as_type_ptr<op::LoopEnd>(expr_it->get()->get_node())) {
            const auto begin_it =
                linear_ir.find_before(expr_it, linear_ir.get_expr_by_node(loop_end->get_loop_begin()));
            const auto end_it = expr_it;
            OPENVINO_ASSERT(decompose(linear_ir, begin_it, end_it, loop_end),
                            "Loop with ID ",
                            loop_end->get_id(),
                            " has not been decomposed!");
            modified = true;
        }
    }
    // Expressions are iterated and check if it's connected to
    // result and replace with new one. The first such expression may not connect to the first result in
    // m_result_expressions. This means the result replacement doesn't happen in order. Result replacement push_back
    // result to m_result_expressions. So Result order could be changed after this pass. We need to sort results again.
    if (modified) {
        linear_ir.sort_results();
    }

    return modified;
}

}  // namespace ov::snippets::lowered::pass
