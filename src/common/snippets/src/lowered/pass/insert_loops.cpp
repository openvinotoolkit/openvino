// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_loops.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

using LoopPort = LinearIR::LoopManager::LoopPort;

namespace {
std::vector<size_t> get_outer_loop_ids(const ExpressionPtr& expr, size_t loop_id) {
    const auto loop_ids = expr->get_loop_ids();
    const auto it = std::find(loop_ids.cbegin(), loop_ids.cend(), loop_id);
    OPENVINO_ASSERT(it != loop_ids.cend(), "Loop ID hasn't been found");
    return std::vector<size_t>(loop_ids.cbegin(), it);
}
}  // namespace

InsertLoops::InsertLoops() : Pass() {}

void InsertLoops::filter_ports(std::vector<LoopPort>& loop_entries, std::vector<LoopPort>& loop_exits) {
    std::vector<LoopPort> new_loop_entries;
    std::vector<LoopPort> new_loop_exits;
    new_loop_entries.reserve(loop_entries.size());
    new_loop_exits.reserve(loop_exits.size());

    for (const auto& loop_entry_point : loop_entries) {
        const auto& expr = loop_entry_point.expr_port->get_expr();
        const auto ma = ov::as_type_ptr<op::MemoryAccess>(expr->get_node());
        if (ma && ma->is_memory_access_input_port(loop_entry_point.expr_port->get_index())) {
            new_loop_entries.push_back(loop_entry_point);
        }
    }

    for (const auto& loop_exit_point : loop_exits) {
        const auto& expr = loop_exit_point.expr_port->get_expr();
        const auto ma = ov::as_type_ptr<op::MemoryAccess>(expr->get_node());
        if (ma && ma->is_memory_access_output_port(loop_exit_point.expr_port->get_index())) {
            new_loop_exits.push_back(loop_exit_point);
        }
    }

    loop_entries = new_loop_entries;
    loop_exits = new_loop_exits;
}

void InsertLoops::insertion(LinearIR& linear_ir, const LinearIR::LoopManagerPtr& loop_manager, size_t loop_id, bool has_outer_loop) {
    const auto loop_info = loop_manager->get_loop_info(loop_id);
    auto loop_entries = loop_info->entry_points;
    auto loop_exits = loop_info->exit_points;
    const auto work_amount = loop_info->work_amount;
    const auto work_amount_increment = loop_info->increment;

    LinearIR::constExprIt loop_begin_pos, loop_end_pos;
    loop_manager->get_loop_bounds(linear_ir, loop_id, loop_begin_pos, loop_end_pos);

    // Remove non MemoryAccess ports since Loop can have only GPR inputs
    filter_ports(loop_entries, loop_exits);

    const auto in_out_num = loop_entries.size() + loop_exits.size();
    std::vector<int64_t> ptr_increments, finalization_offsets, io_data_sizes;
    std::vector<PortConnectorPtr> loop_end_inputs;
    ptr_increments.reserve(in_out_num);
    finalization_offsets.reserve(in_out_num);
    io_data_sizes.reserve(in_out_num);
    loop_end_inputs.reserve(in_out_num);

    auto init_params = [&ptr_increments, &finalization_offsets, &io_data_sizes, &loop_end_inputs](const std::vector<LoopPort>& ports) {
        for (const auto& port : ports) {
            ptr_increments.push_back(port.ptr_increment);
            finalization_offsets.push_back(port.finalization_offset);
            io_data_sizes.push_back(port.data_size);
            loop_end_inputs.push_back(port.expr_port->get_port_connector_ptr());
        }
    };
    init_params(loop_entries);
    init_params(loop_exits);

    const auto& loop_begin = std::make_shared<op::LoopBegin>();
    const auto& loop_begin_expr = linear_ir.create_expression(loop_begin, std::vector<PortConnectorPtr>{});
    linear_ir.insert(loop_begin_pos, loop_begin_expr);

    const auto& loop_end = std::make_shared<op::LoopEnd>(
            loop_begin->output(0), work_amount, work_amount_increment, ptr_increments, finalization_offsets,
            io_data_sizes, loop_entries.size(), loop_exits.size(), loop_id);
    loop_end->has_outer_loop = has_outer_loop;

    // Add LoopBegin port connector
    loop_end_inputs.push_back(loop_begin_expr->get_output_port_connector(0));

    const auto& loop_end_expr = linear_ir.create_expression(loop_end, loop_end_inputs);
    const auto& it = linear_ir.insert(loop_end_pos, loop_end_expr);

    const auto outer_loop_ids = get_outer_loop_ids(*std::prev(it), loop_id);
    loop_begin_expr->set_loop_ids(outer_loop_ids);
    loop_end_expr->set_loop_ids(outer_loop_ids);
}

bool InsertLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertLoops")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();

    std::set<size_t> inserted_loops;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto expr = *expr_it;
        const auto& node = expr->get_node();
        if (ov::is_type<op::LoopBase>(node) ||
            ov::is_type<ov::op::v0::Parameter>(node) ||
            ov::is_type<ov::op::v0::Result>(node))
            continue;

        // Outer Loop ----> Inner Loop
        const auto expr_loops = expr->get_loop_ids();
        const auto loop_depth = expr_loops.size();
        for (size_t i = 0; i < loop_depth; ++i) {
            const auto loop_id = expr_loops[i];
            if (inserted_loops.count(loop_id) == 0) {
                const bool has_outer_loop = i > 0 && inserted_loops.find(expr_loops[i - 1]) != inserted_loops.end();
                insertion(linear_ir, loop_manager, loop_id, has_outer_loop);
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
