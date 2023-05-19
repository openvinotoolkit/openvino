// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/init_loops.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

namespace {
void filter_ports(LinearIR& linear_ir,
                  std::vector<ExpressionPort>& loop_entries, std::vector<ExpressionPort>& loop_exits) {
    std::vector<ExpressionPort> new_loop_entries;
    std::vector<ExpressionPort> new_loop_exits;
    new_loop_entries.reserve(loop_entries.size());
    new_loop_exits.reserve(loop_exits.size());

    for (const auto& loop_entry_point : loop_entries) {
        const auto& expr = loop_entry_point.get_expr();
        const auto ma = ov::as_type_ptr<op::MemoryAccess>(expr->get_node());
        if (ma && ma->is_memory_access_input_port(loop_entry_point.get_index())) {
            new_loop_entries.push_back(loop_entry_point);
        }
    }

    for (const auto& loop_exit_point : loop_exits) {
        const auto& expr = loop_exit_point.get_expr();
        const auto ma = ov::as_type_ptr<op::MemoryAccess>(expr->get_node());
        if (ma && ma->is_memory_access_output_port(loop_exit_point.get_index())) {
            new_loop_exits.push_back(loop_exit_point);
        }
    }

    loop_entries = new_loop_entries;
    loop_exits = new_loop_exits;
}

int64_t get_dim_stride(const size_t dim, const std::vector<size_t>& layout, const std::vector<size_t>& shape) {
    int64_t stride = 1;
    for (int i = static_cast<int>(layout.size()) - 1; i >= 0; i--) {
        if (layout[i] == dim)
            break;
        stride *= static_cast<int64_t>(shape[layout[i]]);
    }
    return stride;
}
}  // namespace

InitLoops::InitLoops() : Pass() {}

std::vector<int64_t> InitLoops::init_ptr_increments(const std::vector<ExpressionPort>& loop_inputs,
                                                    const std::vector<ExpressionPort>& loop_outputs,
                                                    size_t dim_idx) {
     std::vector<int64_t> ptr_increments;
    // Note: Need to find max relevant dim expr to account for broadcasting, collect relevant_dims as well
    size_t max_relevant_dim_size = 1;
    for (const auto& loop_input : loop_inputs) {
        const auto& layout = loop_input.get_descriptor_ptr()->get_layout();
        const auto& shape = loop_input.get_descriptor_ptr()->get_shape();
        const auto& dim = *(layout.rbegin() + dim_idx);
        max_relevant_dim_size = std::max(shape[dim], max_relevant_dim_size);
    }
    for (const auto& loop_output : loop_outputs) {
        const auto& layout = loop_output.get_descriptor_ptr()->get_layout();
        const auto& shape = loop_output.get_descriptor_ptr()->get_shape();
        const auto& dim = *(layout.rbegin() + dim_idx);
        max_relevant_dim_size = std::max(shape[dim], max_relevant_dim_size);
    }

    for (const auto& loop_input : loop_inputs) {
        // For strides we have to use layout from source since source writes data by special rules
        const auto source = *loop_input.get_connected_ports().begin();
        const auto& layout = loop_input.get_descriptor_ptr()->get_layout();
        const auto& shape = loop_input.get_descriptor_ptr()->get_shape();
        const auto& dim = *(layout.rbegin() + dim_idx);
        int64_t ptr_increment = 0;
        // If relevant dim is not broadcasted, then ptr_increment is the dim stride in the new layout
        if (!(shape[dim] == 1 && max_relevant_dim_size != 1))
            ptr_increment = get_dim_stride(dim, source.get_descriptor_ptr()->get_layout(), shape);
        ptr_increments.push_back(ptr_increment);
    }

    for (const auto& loop_output : loop_outputs) {
        const auto& layout = loop_output.get_descriptor_ptr()->get_layout();
        const auto& shape = loop_output.get_descriptor_ptr()->get_shape();
        const auto& dim = *(layout.rbegin() + dim_idx);
        int64_t ptr_increment = 0;
        // If relevant dim is not broadcasted, then ptr_increment is the dim stride in the new layout
        if (!(shape[dim] == 1 && max_relevant_dim_size != 1))
            ptr_increment = get_dim_stride(dim, layout, shape);
        ptr_increments.push_back(ptr_increment);
    }

    return ptr_increments;
}

std::vector<int64_t> InitLoops::init_finalization_offsets(const std::vector<int64_t>& ptr_increments, size_t work_amount) {
    std::vector<int64_t> finalization_offsets;
    for (const auto& ptr_incr : ptr_increments) {
        int64_t offset = -1 * ptr_incr * work_amount;
        finalization_offsets.push_back(offset);
    }
    return finalization_offsets;
}

std::vector<int64_t> InitLoops::init_element_type_sizes(const std::vector<ExpressionPort>& loop_inputs,
                                                        const std::vector<ExpressionPort>& loop_outputs) {
    std::vector<int64_t> element_types;
    element_types.reserve(loop_inputs.size() + loop_outputs.size());
    for (const auto& in : loop_inputs) {
        element_types.push_back(in.get_expr()->get_node()->get_input_element_type(in.get_index()).size());
    }
    for (const auto& out : loop_outputs) {
        element_types.push_back(out.get_expr()->get_node()->get_output_element_type(out.get_index()).size());
    }
    return element_types;
}

void InitLoops::insertion(LinearIR& linear_ir, const LinearIR::LoopManager::LoopInfoPtr& loop_info,
                          size_t loop_id, size_t dim_idx, bool has_outer_loop) {
    auto loop_entries = loop_info->entry_exprs;
    auto loop_exits = loop_info->exit_exprs;
    const auto work_amount = loop_info->work_amount;
    const auto work_amount_increment = loop_info->increment;

    LinearIR::constExprIt loop_begin_pos, loop_end_pos;
    LinearIR::LoopManager::get_loop_bounds(linear_ir, loop_entries, loop_exits, loop_begin_pos, loop_end_pos, loop_id);

    filter_ports(linear_ir, loop_entries, loop_exits);

    auto ptr_increments = init_ptr_increments(loop_entries, loop_exits, dim_idx);
    auto finalization_offsets = init_finalization_offsets(ptr_increments, work_amount);
    const auto io_data_sizes = init_element_type_sizes(loop_entries, loop_exits);

    const auto& loop_begin = std::make_shared<op::LoopBegin>();
    const auto& loop_begin_expr = linear_ir.create_expression(loop_begin, std::vector<PortConnectorPtr>{});
    linear_ir.insert(loop_begin_pos, loop_begin_expr);

    const auto& loop_end = std::make_shared<op::LoopEnd>(
            loop_begin->output(0), work_amount, work_amount_increment, ptr_increments, finalization_offsets,
            io_data_sizes, loop_entries.size(), loop_exits.size());
    loop_end->has_outer_loop = has_outer_loop;

    std::vector<PortConnectorPtr> loop_end_inputs;
    for (const auto& expr_port : loop_entries)
        loop_end_inputs.push_back(expr_port.get_expr()->get_input_port_connector(expr_port.get_index()));
    for (const auto& expr_port : loop_exits)
        loop_end_inputs.push_back(expr_port.get_expr()->get_output_port_connector(expr_port.get_index()));
    loop_end_inputs.push_back(loop_begin_expr->get_output_port_connector(0));

    const auto& loop_end_expr = linear_ir.create_expression(loop_end, loop_end_inputs);
    linear_ir.insert(loop_end_pos, loop_end_expr);
}

bool InitLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InitLoops")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();

    std::set<size_t> inserted_loops;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto expr = *expr_it;
        const auto& node = expr->get_node();
        if (ov::is_type<op::LoopBase>(node) ||
            ov::is_type<op::Buffer>(node) ||     // Need to cover Buffer
            ov::is_type<ov::op::v0::Parameter>(node) ||
            ov::is_type<ov::op::v0::Result>(node))
            continue;

        // Outer Loop ----> Inner Loop
        const auto expr_loops = expr->get_loop_ids();
        const auto loop_depth = expr_loops.size();
        for (size_t i = 0; i < loop_depth; ++i) {
            const auto loop_id = expr_loops[i];
            if (loop_id == Expression::LOOP_NULL_ID)
                continue;
            bool need_to_insert = inserted_loops.find(loop_id) == inserted_loops.end();
            if (need_to_insert) {
                const auto loop_info = loop_manager->get_loop_info(loop_id);
                const bool has_outer_loop = i > 0 && inserted_loops.find(expr_loops[i - 1]) != inserted_loops.end();
                insertion(linear_ir, loop_info, loop_id, loop_depth - i - 1, has_outer_loop);
                inserted_loops.insert(loop_id);  // save Loop ID
            }
        }
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
