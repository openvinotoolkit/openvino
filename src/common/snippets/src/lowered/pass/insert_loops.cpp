// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_loops.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

void InsertLoops::insertion(LinearIR& linear_ir, const LoopManagerPtr& loop_manager, size_t loop_id) {
    const auto loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(loop_id);
    const auto work_amount = loop_info->get_work_amount();
    const auto work_amount_increment = loop_info->get_increment();
    const auto in_num = loop_info->get_input_count();
    const auto out_num = loop_info->get_output_count();

    std::vector<PortConnectorPtr> loop_end_inputs;
    loop_end_inputs.reserve(in_num + out_num);
    loop_info->iterate_through_ports([&loop_end_inputs](const LoopPort& port) {
        loop_end_inputs.push_back(port.get_expr_port()->get_port_connector_ptr());
    });

    const auto is_incremented = loop_info->get_is_incremented();
    const auto ptr_increments = loop_info->get_ptr_increments();
    const auto finalization_offsets = loop_info->get_finalization_offsets();
    const auto io_data_sizes = loop_info->get_data_sizes();

    const auto loop_begin = std::make_shared<op::LoopBegin>();
    const auto loop_end = std::make_shared<op::LoopEnd>(loop_begin, work_amount, work_amount_increment, is_incremented, ptr_increments,
                                                        finalization_offsets, io_data_sizes, in_num, out_num, loop_id);

    const auto loop_bounds = loop_manager->get_loop_bounds(linear_ir, loop_id);
    const auto outer_loop_ids = loop_manager->get_outer_expr_loops(*loop_bounds.first, loop_id);

    const auto loop_begin_expr = *linear_ir.insert_node(loop_begin, std::vector<PortConnectorPtr>{}, outer_loop_ids, false, loop_bounds.first);
    // Add LoopBegin port connector
    loop_end_inputs.push_back(loop_begin_expr->get_output_port_connector(0));
    linear_ir.insert_node(loop_end, loop_end_inputs, outer_loop_ids, false, loop_bounds.second);
}

bool InsertLoops::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertLoops")
    const auto& loop_manager = linear_ir.get_loop_manager();

    std::set<size_t> inserted_loops;
    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto expr = *expr_it;
        const auto& node = expr->get_node();
        if (ov::is_type_any_of<op::LoopBase, ov::op::v0::Parameter, ov::op::v0::Result>(node))
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
