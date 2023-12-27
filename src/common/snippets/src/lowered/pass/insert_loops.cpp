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

void InsertLoops::insertion(LinearIR& linear_ir, const LinearIR::LoopManagerPtr& loop_manager, size_t loop_id) {
    const auto loop_info = loop_manager->get_loop_info(loop_id);
    auto loop_entries = loop_info->get_entry_points();
    auto loop_exits = loop_info->get_exit_points();
    const auto work_amount = loop_info->get_work_amount();
    const auto work_amount_increment = loop_info->get_increment();

    const auto loop_bounds = loop_manager->get_loop_bounds(linear_ir, loop_id);

    const auto in_out_num = loop_entries.size() + loop_exits.size();
    std::vector<bool> is_incremented;
    std::vector<int64_t> ptr_increments, finalization_offsets, io_data_sizes;
    std::vector<PortConnectorPtr> loop_end_inputs;
    is_incremented.reserve(in_out_num);
    ptr_increments.reserve(in_out_num);
    finalization_offsets.reserve(in_out_num);
    io_data_sizes.reserve(in_out_num);
    loop_end_inputs.reserve(in_out_num);

    auto init_params = [&](const std::vector<LoopPort>& ports) {
        for (const auto& port : ports) {
            is_incremented.push_back(port.is_incremented);
            ptr_increments.push_back(port.ptr_increment);
            finalization_offsets.push_back(port.finalization_offset);
            io_data_sizes.push_back(port.data_size);
            loop_end_inputs.push_back(port.expr_port->get_port_connector_ptr());
        }
    };
    init_params(loop_entries);
    init_params(loop_exits);

    // Should be inited by LoopInfo
    const auto is_dynamic_loop = false;

    const auto outer_loop_ids = loop_manager->get_outer_expr_loops(*loop_begin_pos, loop_id);

    const auto& loop_begin = make_loop_begin(is_dynamic_loop);
    const auto loop_begin_expr = *linear_ir.insert_node(loop_begin, std::vector<PortConnectorPtr>{}, outer_loop_ids, false, loop_bounds.first);

    const auto& loop_end = make_loop_end(
            is_dynamic_loop, loop_begin->output(0), work_amount, work_amount_increment, is_incremented, ptr_increments,
            finalization_offsets, io_data_sizes, loop_entries.size(), loop_exits.size(), loop_id);
    // Add LoopBegin port connector
    loop_end_inputs.push_back(loop_begin_expr->get_output_port_connector(0));
    linear_ir.insert_node(loop_end, loop_end_inputs, outer_loop_ids, false, loop_bounds.second);
}

std::shared_ptr<op::LoopBegin> InsertLoops::make_loop_begin(bool is_dynamic) {
    if (is_dynamic)
        return std::make_shared<op::LoopBeginDynamic>();
    return std::make_shared<op::LoopBeginStatic>();
}

std::shared_ptr<op::LoopEnd> InsertLoops::make_loop_end(bool is_dynamic, const Output<Node>& loop_begin, size_t work_amount, size_t work_amount_increment,
                                                        std::vector<bool> is_incremented, std::vector<int64_t> ptr_increments,
                                                        std::vector<int64_t> finalization_offsets, std::vector<int64_t> element_type_sizes,
                                                        size_t input_num, size_t output_num, size_t id) {
    if (is_dynamic)
        return std::make_shared<op::LoopEndDynamic>(loop_begin, work_amount, work_amount_increment, is_incremented, ptr_increments, finalization_offsets,
                                                    element_type_sizes, input_num, output_num, id);
    return std::make_shared<op::LoopEndStatic>(loop_begin, work_amount, work_amount_increment, is_incremented, ptr_increments, finalization_offsets,
                                               element_type_sizes, input_num, output_num, id);
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
        const auto& expr_loops = expr->get_loop_ids();
        const auto loop_depth = expr_loops.size();
        for (size_t i = 0; i < loop_depth; ++i) {
            const auto loop_id = expr_loops[i];
            if (inserted_loops.count(loop_id) == 0) {
                insertion(linear_ir, loop_manager, loop_id);
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
