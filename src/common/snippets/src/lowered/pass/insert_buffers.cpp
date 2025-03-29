// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_buffers.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"


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
}  // namespace

LinearIR::constExprIt InsertBuffers::insertion_position(const LinearIR& linear_ir, const LoopManagerPtr& loop_manager,
                                                        const ExpressionPtr& up_expr, const ExpressionPtr& down_expr) {
    const auto& up_loops = up_expr->get_loop_ids();
    const auto& down_loops = down_expr->get_loop_ids();
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
    // If upper and lower expressions are in the same loop, we should insert Buffer between them
    if (loop_idx == up_loop_count && loop_idx == down_loop_count) {
        return linear_ir.find(down_expr);
    }
    OPENVINO_THROW("Incorrect configuration for Buffer insertion!");
}

void InsertBuffers::insertion(LinearIR& linear_ir,
                              const LinearIR::constExprIt& begin_it,
                              const LinearIR::constExprIt& end_it,
                              const LoopManagerPtr& loop_manager,
                              const std::vector<ExpressionPort>& loop_entries,
                              const std::vector<ExpressionPort>& loop_exits) const {
    for (const auto& entry_port : loop_entries) {
        const auto& expr = entry_port.get_expr();
        const auto port_idx = entry_port.get_index();
        const auto node = expr->get_node();
        auto parent_expr_output = expr->get_input_port_connector(port_idx)->get_source();
        auto parent_expr = parent_expr_output.get_expr();
        bool has_shape_infer_parent = false;
        auto top_shape_infer_expr = expr;
        // parent before shape infer ops is used to determine if buffer needed according loopInfo
        const auto& shape_infer_parents = utils::get_first_parent_shape_infer_expr_seq(parent_expr);
        if (!shape_infer_parents.empty()) {
            parent_expr_output = shape_infer_parents.back()->get_input_port_connector(0)->get_source();
            has_shape_infer_parent = true;
            top_shape_infer_expr = shape_infer_parents.back();
            parent_expr = parent_expr_output.get_expr();
        }
        const auto& parent_port = parent_expr_output.get_index();
        const auto& parent = parent_expr->get_node();
        if (ov::is_type_any_of<op::Buffer,
                               op::VectorBuffer,
                               ov::op::v0::Parameter,
                               ov::op::v0::Constant,
                               op::RankNormalization>(parent))
            continue;

        // Each MemoryAccess op needs Buffer
        const auto parent_ma = std::dynamic_pointer_cast<modifier::MemoryAccess>(parent);
        const auto node_ma = std::dynamic_pointer_cast<modifier::MemoryAccess>(node);
        bool is_buffer_needed = (parent_ma && parent_ma->is_memory_access_output_port(parent_port)) ||
                                (node_ma && node_ma->is_memory_access_input_port(port_idx));
        const auto& current_loops = expr->get_loop_ids();
        const auto& parent_loops = parent_expr->get_loop_ids();
        const auto buffer_loop_ids = get_buffer_loop_ids(current_loops, parent_loops, is_buffer_needed);

        if (is_buffer_needed) {
            // We should insert Buffer between first different Loops.
            // Example: Target Parent Loop identifies: 3, 2, 1
            //          Current expr Loop identifies:  3, 4, 6
            //          Need to insert between 2nd and 4th Loops - after 2nd Loop
            const auto pos = insertion_position(linear_ir, loop_manager, parent_expr, expr);
            const auto buffer = std::make_shared<op::Buffer>(parent->output(parent_port));
            const auto buffer_consumer = has_shape_infer_parent ? top_shape_infer_expr->get_input_port(0)  : entry_port;
            linear_ir.insert_node(buffer, std::vector<ExpressionPort>{ parent_expr_output }, buffer_loop_ids, false, pos, { buffer_consumer  });
        }
    }

    for (const auto& exit_port : loop_exits) {
        const auto& expr = exit_port.get_expr();
        const auto port_idx = exit_port.get_index();
        const auto node = expr->get_node();
        const auto output_connector = exit_port.get_port_connector_ptr();
        const auto child_exprs_inputs = output_connector->get_consumers();
        const auto& current_loops = expr->get_loop_ids();

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
            const auto child_ma = std::dynamic_pointer_cast<modifier::MemoryAccess>(child);
            const auto node_ma = std::dynamic_pointer_cast<modifier::MemoryAccess>(node);
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
                    replace_input_port_connectors(buffer_consumers_inputs, output_connector);
                    potential_consumers.insert(buffer_consumers_inputs.begin(), buffer_consumers_inputs.end());
                    linear_ir.erase(linear_ir.find_after(begin_it, buffer));
                }
            }

            // potential_consumers is unsorted by linear IR set.
            // We have to find first expr in Linear IR from the set to insert Buffer before *all* consumers
            OPENVINO_ASSERT(!potential_consumers.empty(), "Buffer should have one consumer at least");
            const auto& consumer_expr = std::min_element(potential_consumers.begin(), potential_consumers.end(),
                                                         [](const ExpressionPort& l, const ExpressionPort& r) {
                                                             return l.get_expr()->get_exec_num() < r.get_expr()->get_exec_num();
                                                         })->get_expr();

            // We should insert Buffer between first different Loops.
            // Example: Current expr Loop identifies: 3, 2, 1
            //          Target consumers Loop identifies:  3, 4, 6
            //          Need to insert after 2nd Loops
            // Note: All potential consumers must have the same count of first equal Loop identifies and the same count of different last identifies
            const auto pos = insertion_position(linear_ir, loop_manager, expr, consumer_expr);

            auto buffer = std::make_shared<op::Buffer>(node->output(port_idx));
            // We cannot insert Node output connector on Buffer output because not all consumers of Node needs Buffer
            //  Example:
            //       Add
            //      /   \  <- It should be the same PortConnector
            //  Result   Buffer
            //             |    <- It should be new PortConnector
            //            Relu
            // Output port connector is automatically filled from PortDescriptor
            linear_ir.insert_node(buffer, std::vector<ExpressionPort>{ exit_port }, buffer_loop_ids, false, pos, { potential_consumers });
        }
    }
}

bool InsertBuffers::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertBuffers")
    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto loop_data_map = loop_manager->get_map();
    for (const auto& loop_data : loop_data_map) {
        const auto loop_info = loop_data.second;
        const auto loop_entries = loop_info->get_input_ports();
        const auto loop_exits = loop_info->get_output_ports();

        auto cvt_to_expr_ports = [](const std::vector<LoopPort>& loop_ports) {
            std::vector<ExpressionPort> expr_ports(loop_ports.size());
            std::transform(loop_ports.cbegin(), loop_ports.cend(), expr_ports.begin(),
                           [](const LoopPort& loop_port) { return *loop_port.get_expr_port(); });
            return expr_ports;
        };
        // using begin() as expr_it because we work with LoopInfo, not expressions in Linear IR
        insertion(linear_ir, begin, end, loop_manager, cvt_to_expr_ports(loop_entries), cvt_to_expr_ports(loop_exits));
    }

    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto expr = *expr_it;
        const auto node = (*expr_it)->get_node();
        const auto ma = std::dynamic_pointer_cast<modifier::MemoryAccess>(node);
        if (!ma)
            continue;

        const auto input_ports = ma->get_memory_access_input_ports();
        const auto output_ports = ma->get_memory_access_output_ports();
        std::vector<ExpressionPort> loop_entries(input_ports.size()), loop_exits(output_ports.size());
        for (const auto& p : input_ports) {
            loop_entries[p.first] = expr->get_input_port(p.first);
        }
        for (const auto& p : output_ports) {
            loop_exits[p.first] = expr->get_output_port(p.first);
        }

        insertion(linear_ir, expr_it, end, loop_manager, loop_entries, loop_exits);
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
