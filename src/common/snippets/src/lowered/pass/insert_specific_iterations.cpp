// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_specific_iterations.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/linear_ir_builder.hpp"
#include "snippets/lowered/specific_loop_iter_types.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/itt.hpp"

#include <array>

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

namespace {
void connect_cloned_body_with_buffers_outside(LinearIR::constExprIt cur_begin, LinearIR::constExprIt cur_end,
                                              LinearIR::constExprIt res_begin, LinearIR::constExprIt res_end,
                                              LinearIR& linear_ir) {
    for (auto result_it = res_begin, original_it = cur_begin; result_it != res_end; ++result_it, ++original_it) {
        const auto& result_expr = *result_it;
        const auto& original_expr = *original_it;
        // Buffer input can be connected only to outputs of MA ops
        if (std::dynamic_pointer_cast<modifier::MemoryAccess>(original_expr->get_node())) {
            for (size_t i = 0; i < original_expr->get_output_count(); i++) {
                const auto& consumers = original_expr->get_output_port_connector(i)->get_consumers();
                for (const auto& consumer : consumers) {
                    const auto consumer_expr = consumer.get_expr();
                    const auto buffer_expr = ov::as_type_ptr<BufferExpression>(consumer_expr);
                    if (buffer_expr && std::find(cur_begin, cur_end, consumer.get_expr()) == cur_end) {
                        std::vector<PortDescriptorPtr> new_descs = {buffer_expr->get_input_port_descriptor(consumer.get_index())->clone()};
                        std::vector<PortConnectorPtr> new_inputs = {result_expr->get_output_port_connector(i)};
                        OutputVector new_op_inputs = {result_expr->get_node()->output(i)};
                        for (size_t j = 0; j < buffer_expr->get_input_count(); ++j) {
                            const auto& source = buffer_expr->get_input_port_connector(j)->get_source();
                            new_op_inputs.push_back(source.get_expr()->get_node()->output(source.get_index()));
                            new_descs.push_back(buffer_expr->get_input_port_descriptor(j)->clone());
                            new_inputs.push_back(buffer_expr->get_input_port_connector(j));
                        }
                        const auto new_buffer_op = buffer_expr->get_node()->clone_with_new_inputs(new_op_inputs);
                        linear_ir.replace_with_expr({consumer_expr}, buffer_expr->clone_with_new_inputs(new_buffer_op, new_inputs, new_descs));
                        break;
                    }
                }
            }
        }
    }
}
}  // namespace

bool InsertSpecificIterations::is_decomposed_loop_needed(const UnifiedLoopInfoPtr& unified_loop_info, SpecificLoopIterType type,
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
size_t InsertSpecificIterations::get_decomposed_loop_work_amount(const UnifiedLoopInfoPtr& unified_loop_info, SpecificLoopIterType type,
                                                                 size_t remaining_work_amount) {
    OPENVINO_ASSERT(unified_loop_info, "UnifiedLoopInfo is missed!");
    const auto increment = unified_loop_info->get_increment();
    const auto is_dynamic = utils::is_dynamic_value(remaining_work_amount);

    switch (type) {
        case (SpecificLoopIterType::FIRST_ITER):
            // We don't set always `increment` for first iterations since in dynamic `work_amount` can be less than `increment`
            return is_dynamic ? remaining_work_amount : increment;
        case (SpecificLoopIterType::MAIN_BODY):
            return is_dynamic ? remaining_work_amount : (remaining_work_amount / increment) * increment;
        case (SpecificLoopIterType::LAST_ITER): {
            OPENVINO_ASSERT(is_dynamic || remaining_work_amount < unified_loop_info->get_increment(),
                            "Last iter work amount (", remaining_work_amount,
                            ") must be less than the UnifiedLoopInfo's increment: ", unified_loop_info->get_increment());
            return remaining_work_amount;
        }
        default:
            OPENVINO_THROW("Unknown SpecificLoopIterType!");
    }
}
size_t InsertSpecificIterations::get_decomposed_loop_increment(const UnifiedLoopInfoPtr& unified_loop_info, SpecificLoopIterType type,
                                                               size_t remaining_work_amount) {
    OPENVINO_ASSERT(unified_loop_info, "UnifiedLoopInfo is missed!");
    const auto increment = unified_loop_info->get_increment();

    switch (type) {
        case (SpecificLoopIterType::FIRST_ITER):
        case (SpecificLoopIterType::MAIN_BODY):
            return increment;
        case(SpecificLoopIterType::LAST_ITER):
            return remaining_work_amount;
        default:
            OPENVINO_THROW("Unknown SpecificLoopIterType!");
    }
}

LoopManager::LoopBounds InsertSpecificIterations::insert_copy_loop(LinearIR& linear_ir, const size_t loop_id, const LinearIR::constExprIt& insert_pos,
                                                                   std::vector<LoopPort>& new_entry_ports, std::vector<LoopPort>& new_exit_ports) {
    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto loop_bounds = loop_manager->get_loop_bounds(linear_ir, loop_id);
    const auto loop_begin_pos = loop_bounds.first;
    const auto loop_end_pos = loop_bounds.second;

    ExpressionMap expression_map;
    const auto& cloning_config = LinearIRBuilder::Config(false);
    const auto& loop_copy_range = LinearIRBuilder(cloning_config).clone_range(loop_begin_pos, std::next(loop_end_pos), expression_map);
    const auto new_loop_begin_pos = linear_ir.insert(insert_pos, loop_copy_range.begin(), loop_copy_range.end());
    const auto new_loop_end_pos = std::prev(insert_pos);

    // Add connections between output of cloned bodies and Buffers from the current LinearIR (Buffers are connections between Loops)
    connect_cloned_body_with_buffers_outside(loop_begin_pos, loop_end_pos, new_loop_begin_pos, new_loop_end_pos, linear_ir);

    auto clone_ports = [&expression_map](const std::vector<LoopPort>& ports, std::vector<LoopPort>& new_ports) {
        new_ports.resize(ports.size());
        for (size_t i = 0; i < ports.size(); ++i) {
            const auto& port = ports[i];
            new_ports[i] = *port.clone_with_new_expr(expression_map[port.expr_port->get_expr().get()]);
        }
    };
    const auto original_loop_info = loop_manager->get_loop_info(loop_id);
    clone_ports(original_loop_info->get_input_ports(), new_entry_ports);
    clone_ports(original_loop_info->get_output_ports(), new_exit_ports);

    return { new_loop_begin_pos, new_loop_end_pos };
}

void InsertSpecificIterations::init_decomposed_loop(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end,
                                                    const ExpandedLoopInfoPtr& decomposed_loop_info, size_t unified_loop_id,
                                                    const std::shared_ptr<op::LoopEnd>& decomposed_loop_end) {
    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto new_id = loop_manager->replace_with_new_loop(linear_ir, begin, std::next(end), decomposed_loop_info, unified_loop_id);
    decomposed_loop_end->set_id(new_id);
    decomposed_loop_end->set_work_amount(decomposed_loop_info->get_work_amount());
    decomposed_loop_end->set_increment(decomposed_loop_info->get_increment());
    decomposed_loop_end->set_ptr_increments(decomposed_loop_info->get_ptr_increments());
    decomposed_loop_end->set_finalization_offsets(decomposed_loop_info->get_finalization_offsets());
    // Note: handlers must be run on the range started with the first operation in the loop body.
    const auto handlers = decomposed_loop_info->get_handler_passes();
    handlers.run(linear_ir, std::next(begin), end);
}

bool InsertSpecificIterations::decompose(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end,
                                         const std::shared_ptr<op::LoopEnd>& loop_end) {
    const auto loop_id = loop_end->get_id();
    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& unified_loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(loop_id);

    auto remaining_work_amount = unified_loop_info->get_work_amount();
    const auto is_wa_dynamic = utils::is_dynamic_value(remaining_work_amount);

    static constexpr std::array<SpecificLoopIterType, 3> loop_iterations = {
        SpecificLoopIterType::FIRST_ITER, SpecificLoopIterType::MAIN_BODY, SpecificLoopIterType::LAST_ITER
    };

    auto decomposed = false;
    for (const auto& iter_type : loop_iterations) {
        if (is_decomposed_loop_needed(unified_loop_info, iter_type, remaining_work_amount)) {
            const auto work_amount = get_decomposed_loop_work_amount(unified_loop_info, iter_type, remaining_work_amount);
            const auto increment = get_decomposed_loop_increment(unified_loop_info, iter_type, remaining_work_amount);
            // Update remaining Loop work amount
            // Note: if work_amount is unknown and increment = 1, it means that a loop will iterate by whole work_amount
            if (!is_wa_dynamic || increment == 1 || iter_type == SpecificLoopIterType::LAST_ITER) {
                remaining_work_amount -= work_amount;
            }

            auto decomposed_loop_end = loop_end;
            auto decomposed_loop_begin_it = begin, decomposed_loop_end_it = end;
            auto decomposed_loop_entry_ports = unified_loop_info->get_input_ports();
            auto decomposed_loop_exit_ports = unified_loop_info->get_output_ports();
            auto decomposed_ptr_increments = unified_loop_info->get_ptr_increments();
            auto decomposed_finalization_offsets = unified_loop_info->get_finalization_offsets();
            auto decomposed_data_sizes = unified_loop_info->get_data_sizes();
            // Need to copy body if there are other specific sup-loops
            // Otherwise we should update the current body
            if (remaining_work_amount > 0) {
                std::tie(decomposed_loop_begin_it, decomposed_loop_end_it) =
                    insert_copy_loop(linear_ir, loop_id, begin, decomposed_loop_entry_ports, decomposed_loop_exit_ports);
                decomposed_loop_end = ov::as_type_ptr<op::LoopEnd>(decomposed_loop_end_it->get()->get_node());
                OPENVINO_ASSERT(decomposed_loop_end, "Cloned Loop does not contain LoopEnd op at the expected place.");

                // Only latest loop iterations must have summarized finalization offsets!
                // Since we inserted copy before the latest iterations, these copies should have reseted offsets
                std::for_each(decomposed_finalization_offsets.begin(), decomposed_finalization_offsets.end(), [](int64_t& offset) {
                    if (!utils::is_dynamic_value(offset))
                        offset = 0;
                });
            }

            const auto decomposed_loop_info = std::make_shared<ExpandedLoopInfo>(work_amount, increment,
                                                                                 decomposed_loop_entry_ports, decomposed_loop_exit_ports,
                                                                                 decomposed_ptr_increments, decomposed_finalization_offsets,
                                                                                 decomposed_data_sizes, iter_type, unified_loop_info);
            init_decomposed_loop(linear_ir, decomposed_loop_begin_it, decomposed_loop_end_it, decomposed_loop_info, loop_id, decomposed_loop_end);

            decomposed = true;
        }
    }

    return decomposed;
}

bool InsertSpecificIterations::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertSpecificIterations")

    bool modified = false;
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        if (const auto loop_end = ov::as_type_ptr<op::LoopEnd>(expr_it->get()->get_node())) {
            const auto begin_it = linear_ir.find_before(expr_it, linear_ir.get_expr_by_node(loop_end->get_loop_begin()));
            const auto end_it = expr_it;
            OPENVINO_ASSERT(decompose(linear_ir, begin_it, end_it, loop_end), "Loop with ID ", loop_end->get_id(), " has not been decomposed!");
            modified = true;
        }
    }

    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
