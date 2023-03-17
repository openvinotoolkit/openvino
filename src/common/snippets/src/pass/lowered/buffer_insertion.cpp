// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/buffer_insertion.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"


namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

BufferInsertion::BufferInsertion(int32_t buffer_allocation_rank)
    : LinearIRTransformation(), m_buffer_allocation_rank(buffer_allocation_rank) {}

LoweredExprIR::constExprIt BufferInsertion::insertion_position(const LoweredExprIR& linear_ir, const LoweredExprIR::LoweredLoopManagerPtr& loop_manager,
                                                               const LoweredExprPtr& up_expr, const LoweredExprPtr& down_expr) {
    if (ov::is_type<op::Brgemm>(up_expr->get_node())) {
        return std::next(std::find(linear_ir.begin(), linear_ir.end(), up_expr));
    } else if (ov::is_type<op::Brgemm>(down_expr->get_node())) {
        return std::find(linear_ir.begin(), linear_ir.end(), down_expr);
    }

    const auto up_loops = up_expr->get_loop_ids();
    const auto down_loops = down_expr->get_loop_ids();
    OPENVINO_ASSERT(up_loops.size() == down_loops.size(), "The Loop IDs must be normalized!");
    size_t loop_idx = 0;
    for (; loop_idx < up_loops.size(); ++loop_idx) {
        if (up_loops[loop_idx] != down_loops[loop_idx])
            break;
    }
    OPENVINO_ASSERT(loop_idx != up_loops.size(), "A Buffer must be inserted only between Loops!");
    const auto loop_id = up_loops[loop_idx];
    const auto loop_info = loop_manager->get_loop_info(loop_id);
    LoweredExprIR::constExprIt loop_begin_pos, loop_end_pos;
    loop_manager->get_loop_bounds(linear_ir, loop_id, loop_begin_pos, loop_end_pos);
    return loop_end_pos;
}

void BufferInsertion::insertion(LoweredExprIR& linear_ir, const LoweredExprIR::LoweredLoopManagerPtr& loop_manager, size_t loop_id,
                                const std::vector<LoweredExprPort>& loop_entries, const std::vector<LoweredExprPort>& loop_exits) {
    for (const auto& entry_point : loop_entries) {
        const auto expr = entry_point.expr;
        const auto port = entry_point.port;
        const auto node = expr->get_node();
        const auto input_td = expr->get_inputs()[port];
        const auto parent_expr_output = linear_ir.get_expr_by_output(input_td);
        const auto& parent_expr = parent_expr_output.expr;
        const auto parent_port = parent_expr_output.port;
        const auto parent = parent_expr->get_node();
        if (ov::is_type<op::Buffer>(parent) ||
            ov::is_type<op::VectorBuffer>(parent) ||
            ov::is_type<opset1::Parameter>(parent) ||
            ov::is_type<opset1::Constant>(parent))
            continue;

        // TODO: Need to cover Brgemm is more pretty
        bool is_buffer_needed = ov::is_type<op::Brgemm>(parent) || ov::is_type<op::Brgemm>(node);
        if (!is_buffer_needed) {
            const auto current_loops = expr->get_loop_ids();
            const auto parent_loops = parent_expr->get_loop_ids();
            const auto current_loop_count = current_loops.size();
            const auto parent_loop_count = parent_loops.size();
            OPENVINO_ASSERT(current_loop_count == parent_loop_count);
            const auto current_loop_lvl = std::distance(current_loops.begin(), std::find(current_loops.begin(), current_loops.end(), loop_id));
            for (size_t i = current_loop_lvl; i < current_loop_count; i++) {
                if (current_loops[i] != parent_loops[i] &&
                    current_loops[i] != LoweredExpr::LOOP_NULL_ID &&
                    parent_loops[i] != LoweredExpr::LOOP_NULL_ID) {
                    is_buffer_needed = true;
                    break;
                }
            }
        }

        if (is_buffer_needed) {
            // We should insert Buffer between first different Loops.
            // Example: Target Parent Loop identifies: 3, 2, 1
            //          Current expr Loop identifies:  3, 4, 6
            //          Need to insert between 2nd and 4th Loops - after 2nd Loop
            const auto pos = insertion_position(linear_ir, loop_manager, parent_expr, expr);
            const auto buffer = std::make_shared<op::Buffer>(parent->output(parent_port), m_buffer_allocation_rank);

            const auto td = std::make_shared<TensorDescriptor>(input_td->get_tensor(),
                                                               input_td->get_subtensor(),
                                                               input_td->get_layout());
            const std::vector<TensorDescriptorPtr> buffer_outs = { td };
            const std::vector<TensorDescriptorPtr> parent_outs = { input_td };
            linear_ir.insert(pos, std::make_shared<LoweredExpr>(buffer, parent_outs, buffer_outs));
            linear_ir.replace_input(expr, port, td);
        }
    }

    for (const auto& exit_point : loop_exits) {
        const auto expr = exit_point.expr;
        const auto port = exit_point.port;
        const auto node = expr->get_node();
        const auto output_td = expr->get_outputs()[port];
        const auto child_exprs_inputs = linear_ir.get_exprs_by_input(output_td);
        const auto current_loops = expr->get_loop_ids();
        const auto current_loop_count = current_loops.size();
        const std::vector<TensorDescriptorPtr> node_outs = {output_td};

        std::set<LoweredExprPort> potential_consumers;
        std::set<LoweredExprPtr> buffers;
        const auto current_loop_lvl = std::distance(current_loops.begin(), std::find(current_loops.begin(), current_loops.end(), loop_id));
        for (const auto& child_expr_input : child_exprs_inputs) {
            const auto child_expr = child_expr_input.expr;
            const auto child = child_expr->get_node();
            if (ov::is_type<opset1::Result>(child))
                continue;
            if (ov::is_type<op::Buffer>(child)) {
                buffers.insert(child_expr);
                continue;
            }
            if (ov::is_type<op::Brgemm>(child) || ov::is_type<op::Brgemm>(node)) {
                potential_consumers.insert(child_expr_input);
                continue;
            }

            const auto child_loops = child_expr->get_loop_ids();
            const auto child_loop_count = child_loops.size();
            OPENVINO_ASSERT(current_loop_count == child_loop_count, "The Loop IDs must be normalized!");
            for (size_t i = current_loop_lvl; i < child_loop_count; i++) {
                if (current_loops[i] != child_loops[i] &&
                    current_loops[i] != LoweredExpr::LOOP_NULL_ID &&
                    child_loops[i] != LoweredExpr::LOOP_NULL_ID) {
                    potential_consumers.insert(child_expr_input);
                    break;
                }
            }
        }

        if (!potential_consumers.empty() || buffers.size() > 1) {
            // If some of children from one common port are different Buffers,
            // we should remove them to insert one common Buffer on one common port
            if (!buffers.empty()) {
                for (const auto& buffer : buffers) {
                    const auto buffer_out = buffer->get_outputs().front();
                    const auto buffer_consumers_inputs = linear_ir.get_exprs_by_input(buffer_out);
                    for (const auto& consumer_input : buffer_consumers_inputs) {
                        const auto consumer = consumer_input.expr;
                        const auto consumer_port = consumer_input.port;
                        linear_ir.replace_input(consumer, consumer_port, output_td);
                    }
                    potential_consumers.insert(buffer_consumers_inputs.begin(), buffer_consumers_inputs.end());
                    linear_ir.erase(std::find(linear_ir.begin(), linear_ir.end(), buffer));
                }
            }

            // We should insert Buffer between first different Loops.
            // Example: Current expr Loop identifies: 3, 2, 1
            //          Target consumers Loop identifies:  3, 4, 6
            //          Need to insert after 2nd Loops
            // Note: All potential consumers must have the same count of first equal Loop identifies and the same count of different last identifies
            // TODO: Need to verify that
            const auto pos = insertion_position(linear_ir, loop_manager, expr, (*potential_consumers.begin()).expr);

            auto buffer = std::make_shared<op::Buffer>(node->output(port), m_buffer_allocation_rank);
            const auto td = std::make_shared<TensorDescriptor>(output_td->get_tensor(),
                                                               output_td->get_subtensor(),
                                                               output_td->get_layout());
            // We cannot insert Node output tensor on Buffer output because not all consumers of Node needs Buffer
            //  Example:
            //       Add
            //      /   \  <- It should be the same TD
            //  Result   Buffer
            //             |    <- It should be new TD
            //            Relu
            const std::vector<TensorDescriptorPtr> buffer_outs = {td};
            linear_ir.insert(pos, std::make_shared<LoweredExpr>(buffer, node_outs, buffer_outs));
            for (const auto& consumer_input : potential_consumers) {
                const auto consumer = consumer_input.expr;
                const auto consumer_port = consumer_input.port;
                linear_ir.replace_input(consumer, consumer_port, td);
            }
        }
    }
}

bool BufferInsertion::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::BufferInsertion")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto loop_data_map = loop_manager->get_map();
    // C++17: for (auto const& [loop_id, loop_info] : loop_data_map)
    for (const auto& loop_data : loop_data_map) {
        const auto loop_id = loop_data.first;
        const auto loop_info = loop_data.second;
        const auto loop_entries = loop_info->entry_exprs;
        const auto loop_exits = loop_info->exit_exprs;
        insertion(linear_ir, loop_manager, loop_id, loop_entries, loop_exits);
    }

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto expr = *expr_it;
        const auto node = (*expr_it)->get_node();
        if (!ov::is_type<op::Brgemm>(node))
            continue;

        std::vector<LoweredExprPort> loop_entries = {LoweredExprPort::make_input(expr, 0),
                                                     LoweredExprPort::make_input(expr, 1)};
        std::vector<LoweredExprPort> loop_exits = {LoweredExprPort::make_output(expr, 0)};

        insertion(linear_ir, loop_manager, LoweredExpr::LOOP_NULL_ID, loop_entries, loop_exits);
    }

    return true;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
