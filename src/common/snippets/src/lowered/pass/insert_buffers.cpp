// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_buffers.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
namespace {
std::vector<size_t> get_buffer_loop_ids(const std::vector<size_t>& lhs, const std::vector<size_t>& rhs, bool& is_buffer_needed) {
    std::vector<size_t> buffer_loop_ids;
    const auto lhs_num = lhs.size();
    const auto rhs_num = rhs.size();
    for (size_t i = 0; i < std::min(lhs_num, rhs_num); ++i) {
        if (lhs[i] == rhs[i]) {
            buffer_loop_ids.push_back(lhs[i]);
            continue;
        }
        is_buffer_needed = true;
        break;
    }
    return buffer_loop_ids;
}

// Ticket: 113744
// TODO: This logic covers only several specific cases so it should be generalized.
ov::Shape compute_allocation_shape(const LinearIR::LoopManagerPtr& loop_manager,
                                   const std::vector<size_t>& buffer_loop_ids,
                                   const std::vector<size_t>& parent_loop_ids,
                                   const ExpressionPort& expr_port,
                                   const int allocation_rank) {
    const auto planar_shape = utils::get_planar_vdims(expr_port);

    const size_t rank = allocation_rank >= 0 ? std::min(static_cast<size_t>(allocation_rank), planar_shape.size()) : planar_shape.size();
    ov::Shape allocation_shape(rank);
    for (size_t i = 0; i < rank; ++i) {
        *(allocation_shape.rbegin() + i) = *(planar_shape.rbegin() + i);
    }

    if (buffer_loop_ids.empty() || parent_loop_ids.empty()) {
        return allocation_shape;
    }

    auto set_rest_dims_to_ones = [&](const int filled_dims_count) {
        for (int i = 0; i < static_cast<int>(allocation_shape.size()) - filled_dims_count; ++i) {
            allocation_shape[i] = 1;
        }
    };

    // In some cases it's possible to allocate less shape
    // 1. Buffer and its parent are in the same loop: allocation size for the outer dimension can be extracted from loop increment
    // 2. Buffer is outside the parent's loops: allocation size can be extracted from the corresponding loop work amount
    // TODO: Use general logic with the help of memory counts for allocation shape computation
    if (buffer_loop_ids.back() == parent_loop_ids.back()) {
        const auto buffer_loop = loop_manager->get_loop_info(buffer_loop_ids.back());
        *(allocation_shape.rbegin() + 1) = buffer_loop->increment;
        set_rest_dims_to_ones(2);
    } else {
        for (size_t i = 0; i < std::min(rank, parent_loop_ids.size()); ++i) {
            const auto loop = loop_manager->get_loop_info(*(parent_loop_ids.rbegin() + i));
            *(allocation_shape.rbegin() + i) = loop->work_amount;
        }
        set_rest_dims_to_ones(static_cast<int>(parent_loop_ids.size()));
    }
    return allocation_shape;
}
}  // namespace

InsertBuffers::InsertBuffers(int32_t buffer_allocation_rank)
    : Pass(), m_buffer_allocation_rank(buffer_allocation_rank) {}

LinearIR::constExprIt InsertBuffers::insertion_position(const LinearIR& linear_ir, const LinearIR::LoopManagerPtr& loop_manager,
                                                        const ExpressionPtr& up_expr, const ExpressionPtr& down_expr) {
    const auto up_loops = up_expr->get_loop_ids();
    const auto down_loops = down_expr->get_loop_ids();
    // If upper expression is out of Loop, we can insert Buffer implicitly after him
    if (up_loops.empty()) {
        return std::next(linear_ir.find(up_expr));
    }
    // If lower expression is out of Loop, we can insert Buffer implicitly before him
    if (down_loops.empty()) {
        return linear_ir.find(down_expr);
    }

    const auto up_loop_count = up_loops.size();
    const auto down_loop_count = down_loops.size();
    size_t loop_idx = 0;
    for (; loop_idx < std::min(up_loop_count, down_loop_count); ++loop_idx) {
        if (up_loops[loop_idx] != down_loops[loop_idx])
            break;
    }
    // If upper expression is inside Loop, we should insert Buffer after this Loop
    if (loop_idx < up_loop_count) {
        const auto up_loop_id = up_loops[loop_idx];
        const auto loop_info = loop_manager->get_loop_info(up_loop_id);
        LinearIR::constExprIt loop_begin_pos, loop_end_pos;
        loop_manager->get_loop_bounds(linear_ir, up_loop_id, loop_begin_pos, loop_end_pos);
        return loop_end_pos;
    }
    // If lower expression is inside Loop, we should insert Buffer before this Loop
    if (loop_idx < down_loop_count) {
        const auto down_loop_id = down_loops[loop_idx];
        const auto loop_info = loop_manager->get_loop_info(down_loop_id);
        LinearIR::constExprIt loop_begin_pos, loop_end_pos;
        loop_manager->get_loop_bounds(linear_ir, down_loop_id, loop_begin_pos, loop_end_pos);
        return loop_begin_pos;
    }
    OPENVINO_THROW("Incorrect configuration for Buffer insertion!");
}

void InsertBuffers::insertion(LinearIR& linear_ir, const LinearIR::constExprIt& expr_it, const LinearIR::LoopManagerPtr& loop_manager,
                              const std::vector<LinearIR::LoopManager::LoopPort>& loop_entries,
                              const std::vector<LinearIR::LoopManager::LoopPort>& loop_exits) {
    for (const auto& entry_point : loop_entries) {
        const auto& entry_port = entry_point.expr_port;
        const auto& expr = entry_port->get_expr();
        const auto port_idx = entry_port->get_index();
        const auto node = expr->get_node();
        const auto& input_connector = expr->get_input_port_connector(port_idx);
        const auto& parent_expr_output = input_connector->get_source();
        const auto& parent_expr = parent_expr_output.get_expr();
        const auto parent_port = parent_expr_output.get_index();
        const auto parent = parent_expr->get_node();
        if (ov::is_type<op::Buffer>(parent) ||
            ov::is_type<op::VectorBuffer>(parent) ||
            ov::is_type<ov::op::v0::Parameter>(parent) ||
            ov::is_type<ov::op::v0::Constant>(parent))
            continue;

        // Each MemoryAccess op needs Buffer
        const auto parent_ma = ov::as_type_ptr<op::MemoryAccess>(parent);
        const auto node_ma = ov::as_type_ptr<op::MemoryAccess>(node);
        bool is_buffer_needed = (parent_ma && parent_ma->is_memory_access_output_port(parent_port)) ||
                                (node_ma && node_ma->is_memory_access_input_port(port_idx));
        const auto current_loops = expr->get_loop_ids();
        const auto parent_loops = parent_expr->get_loop_ids();
        const auto buffer_loop_ids = get_buffer_loop_ids(current_loops, parent_loops, is_buffer_needed);

        if (is_buffer_needed) {
            // We should insert Buffer between first different Loops.
            // Example: Target Parent Loop identifies: 3, 2, 1
            //          Current expr Loop identifies:  3, 4, 6
            //          Need to insert between 2nd and 4th Loops - after 2nd Loop
            const auto pos = insertion_position(linear_ir, loop_manager, parent_expr, expr);
            const auto allocation_shape = compute_allocation_shape(loop_manager,
                                                                   buffer_loop_ids,
                                                                   parent_loops,
                                                                   parent_expr_output,
                                                                   m_buffer_allocation_rank);
            const auto buffer = std::make_shared<op::Buffer>(parent->output(parent_port), allocation_shape);
            PortDescriptorUtils::set_port_descriptor_ptr(buffer->output(0), parent_expr_output.get_descriptor_ptr()->clone());
            // Output connector is automatically filled from PortDescriptor
            const auto buffer_expr = linear_ir.create_expression(buffer, {input_connector});
            linear_ir.insert(pos, buffer_expr);
            linear_ir.replace_input(*entry_port.get(), buffer_expr->get_output_port_connector(0));
            buffer_expr->set_loop_ids(buffer_loop_ids);
        }
    }

    for (const auto& exit_point : loop_exits) {
        const auto& exit_port = exit_point.expr_port;
        const auto& expr = exit_port->get_expr();
        const auto port_idx = exit_port->get_index();
        const auto node = expr->get_node();
        const auto output_connector = exit_port->get_port_connector_ptr();
        const auto child_exprs_inputs = output_connector->get_consumers();
        const auto current_loops = expr->get_loop_ids();
        const std::vector<PortConnectorPtr> node_outs = {output_connector};

        std::set<ExpressionPort> potential_consumers;
        std::set<ExpressionPtr> buffers;
        std::vector<size_t> buffer_loop_ids;
        auto update_buffer_loop_ids = [&buffer_loop_ids, &potential_consumers, &buffers](const std::vector<size_t>& local_ids) {
            if (buffers.empty() && potential_consumers.empty()) {
                buffer_loop_ids = local_ids;
            }
            OPENVINO_ASSERT(local_ids == buffer_loop_ids, "Incorrect loop configuration for Buffers");
        };
        for (const auto& child_expr_input : child_exprs_inputs) {
            const auto& child_expr = child_expr_input.get_expr();
            const auto child_port = child_expr_input.get_index();
            const auto& child = child_expr->get_node();
            if (ov::is_type<ov::op::v0::Result>(child))
                continue;
            if (ov::is_type<op::Buffer>(child)) {
                update_buffer_loop_ids(child_expr->get_loop_ids());
                buffers.insert(child_expr);
                continue;
            }
            // Each MemoryAccess op needs Buffer
            const auto child_ma = ov::as_type_ptr<op::MemoryAccess>(child);
            const auto node_ma = ov::as_type_ptr<op::MemoryAccess>(node);
            bool is_buffer_needed = (child_ma && child_ma->is_memory_access_input_port(child_port)) ||
                                    (node_ma && node_ma->is_memory_access_output_port(port_idx));
            const auto local_buffer_loop_ids = get_buffer_loop_ids(current_loops, child_expr->get_loop_ids(), is_buffer_needed);

            if (is_buffer_needed) {
                update_buffer_loop_ids(local_buffer_loop_ids);
                potential_consumers.insert(child_expr_input);
            }
        }

        if (!potential_consumers.empty() || buffers.size() > 1) {
            // If some of children from one common port are different Buffers,
            // we should remove them to insert one common Buffer on one common port
            if (!buffers.empty()) {
                for (const auto& buffer : buffers) {
                    const auto& buffer_out = buffer->get_output_port_connector(0);
                    const auto buffer_consumers_inputs = buffer_out->get_consumers();
                    linear_ir.replace_input(buffer_consumers_inputs, output_connector);
                    potential_consumers.insert(buffer_consumers_inputs.begin(), buffer_consumers_inputs.end());
                    linear_ir.erase(linear_ir.find_after(expr_it, buffer));
                }
            }

            // potential_consumers is unsorted by linear IR set.
            // We have to find first expr in Linear IR from the set to insert Buffer before *all* consumers
            // [113536]: Remove this logic with `std::find` using, when expression numeration will be supported
            OPENVINO_ASSERT(!potential_consumers.empty(), "Buffer should have one consumer at least");
            auto consumer_expr = potential_consumers.begin()->get_expr();
            if (potential_consumers.size() > 1) {
                std::set<ExpressionPtr> consumers;
                for (const auto& port : potential_consumers)
                    consumers.insert(port.get_expr());
                const auto it = std::find_if(expr_it, linear_ir.cend(),
                                             [&consumers](const ExpressionPtr& expr) { return consumers.count(expr) > 0; });
                OPENVINO_ASSERT(it != linear_ir.cend(), "Consumer of Buffer has not been found in Linear IR");
                consumer_expr = *it;
            }

            // We should insert Buffer between first different Loops.
            // Example: Current expr Loop identifies: 3, 2, 1
            //          Target consumers Loop identifies:  3, 4, 6
            //          Need to insert after 2nd Loops
            // Note: All potential consumers must have the same count of first equal Loop identifies and the same count of different last identifies
            const auto pos = insertion_position(linear_ir, loop_manager, expr, consumer_expr);

            const auto allocation_shape = compute_allocation_shape(loop_manager,
                                                                   buffer_loop_ids,
                                                                   current_loops,
                                                                   *exit_port,
                                                                   m_buffer_allocation_rank);
            auto buffer = std::make_shared<op::Buffer>(node->output(port_idx), allocation_shape);
            PortDescriptorUtils::set_port_descriptor_ptr(buffer->output(0), exit_port->get_descriptor_ptr()->clone());
            // We cannot insert Node output connector on Buffer output because not all consumers of Node needs Buffer
            //  Example:
            //       Add
            //      /   \  <- It should be the same PortConnector
            //  Result   Buffer
            //             |    <- It should be new PortConnector
            //            Relu
            // Output port connector is automatically filled from PortDescriptor
            const auto buffer_expr = linear_ir.create_expression(buffer, node_outs);
            linear_ir.insert(pos, buffer_expr);
            linear_ir.replace_input(potential_consumers, buffer_expr->get_output_port_connector(0));
            buffer_expr->set_loop_ids(buffer_loop_ids);
        }
    }
}

bool InsertBuffers::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertBuffers")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto loop_data_map = loop_manager->get_map();
    for (const auto& loop_data : loop_data_map) {
        const auto loop_info = loop_data.second;
        const auto loop_entries = loop_info->entry_points;
        const auto loop_exits = loop_info->exit_points;
        // using begin() as expr_it because we work with LoopInfo, not expressions in Linear IR
        insertion(linear_ir, linear_ir.cbegin(), loop_manager, loop_entries, loop_exits);
    }

    for (auto expr_it = linear_ir.cbegin(); expr_it != linear_ir.cend(); expr_it++) {
        const auto expr = *expr_it;
        const auto node = (*expr_it)->get_node();
        const auto ma = ov::as_type_ptr<op::MemoryAccess>(node);
        if (!ma)
            continue;

        const auto input_ports = ma->get_memory_access_input_ports();
        const auto output_ports = ma->get_memory_access_output_ports();
        std::vector<LinearIR::LoopManager::LoopPort> loop_entries(input_ports.size()), loop_exits(output_ports.size());
        for (const auto& p : input_ports) {
            loop_entries[p.first] = expr->get_input_port(p.first);
        }
        for (const auto& p : output_ports) {
            loop_exits[p.first] = expr->get_output_port(p.first);
        }

        insertion(linear_ir, expr_it, loop_manager, loop_entries, loop_exits);
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
