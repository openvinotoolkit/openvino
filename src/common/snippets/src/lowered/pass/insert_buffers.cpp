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
    const auto planar_shape = utils::get_preordered_vdims(expr_port);

    const size_t rank = allocation_rank >= 0 ? std::min(static_cast<size_t>(allocation_rank), planar_shape.size()) : planar_shape.size();
    ov::Shape allocation_shape(rank);
    for (size_t i = 0; i < rank; ++i) {
        *(allocation_shape.rbegin() + i) = *(planar_shape.rbegin() + i);
    }

    if (buffer_loop_ids.empty() || parent_loop_ids.empty()) {
        return allocation_shape;
    }

    // If subtensor is set, its information is used for allocation shape computation. Two situations are possible:
    // 1. Buffer is outside the parent loop: the corresponding subtensor value is ignored, parent loop work amount is set instead
    // 2. Buffer is inside the parent loop: the corresponding subtensor value is used in allocation shape.
    // Since we can defenitely know which subtensor value corresponds to the loop only for 1st case
    // (we can extract this info from loop exit port), we copy subtensor, and then replace subtensor values with parent loop work amount if needed.
    // Example:
    // Parent subtensor: [M_blk, N_blk]
    // Buffer loop idces: [M_loop_idx], parent loop idces: [M_loop_idx, N_loop_idx]
    //
    // 1. Allocation shape is set to subtensor: [M_blk, N_blk]
    // 2. Buffer is inside M_loop_idx loop => allocation shape is not changed
    // 3. Buffer is outside N_loop_idx loop => the corresponding allocation shape value is replaced with N loop work amount
    // So the result allocation shape is [M_blk, N_loop_work_amount]
    const auto& subtensor =  expr_port.get_descriptor_ptr()->get_subtensor();
    if (!subtensor.empty()) {
        for (size_t i = 0; i < std::min(rank, subtensor.size()); ++i) {
            auto& cur_val = *(allocation_shape.rbegin() + i);
            const auto& subtensor_val = *(subtensor.rbegin() + i);
            cur_val = std::min(cur_val, subtensor_val);
        }
        for (const auto& parent_loop : parent_loop_ids) {
            if (std::find(buffer_loop_ids.begin(), buffer_loop_ids.end(), parent_loop) == buffer_loop_ids.end()) {
                const auto loop_info = loop_manager->get_loop_info(parent_loop);
                const auto& exit_points = loop_info->get_exit_points();
                auto it = std::find_if(exit_points.begin(),
                                       exit_points.end(),
                                       [&expr_port](const LinearIR::LoopManager::LoopPort& port) {
                                           return *port.expr_port == expr_port;
                                       });
                OPENVINO_ASSERT(it != exit_points.end(), "compute_allocation_shape: exit point of parent loop can not be found");
                const auto& loop_port = *it;
                if (loop_port.is_incremented && loop_port.dim_idx < allocation_shape.size()) {
                    *(allocation_shape.rbegin() + loop_port.dim_idx) = loop_info->get_work_amount();
                }
            }
        }
    } else {
        // WA: In case of empty subtensors another information have to be used to update allocation shape.
        for (size_t i = 0; i < std::min(rank, parent_loop_ids.size()); ++i) {
            const auto loop = loop_manager->get_loop_info(*(parent_loop_ids.rbegin() + i));
            OPENVINO_ASSERT(loop->get_dim_idx() == i, "compute_allocation_shape: eltwise loop has unexpected dimension index");
            *(allocation_shape.rbegin() + i) = loop->get_work_amount();
        }
        for (int i = 0; i < allocation_rank - static_cast<int>(parent_loop_ids.size()); ++i) {
            allocation_shape[i] = 1;
        }
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
        return loop_manager->get_loop_bounds(linear_ir, up_loop_id).second;
    }
    // If lower expression is inside Loop, we should insert Buffer before this Loop
    if (loop_idx < down_loop_count) {
        const auto down_loop_id = down_loops[loop_idx];
        return loop_manager->get_loop_bounds(linear_ir, down_loop_id).first;
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
            const auto buffer = std::make_shared<op::IntermediateMemoryBuffer>(parent->output(parent_port), allocation_shape);
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
            auto buffer = std::make_shared<op::IntermediateMemoryBuffer>(node->output(port_idx), allocation_shape);
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
        const auto loop_entries = loop_info->get_entry_points();
        const auto loop_exits = loop_info->get_exit_points();
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
