// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_loops.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

void InsertLoops::insertion(LinearIR& linear_ir, const LoopManagerPtr& loop_manager, size_t loop_id) {
    const auto loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(loop_id);
    auto loop_entries = loop_info->get_input_ports();
    auto loop_exits = loop_info->get_output_ports();
    const auto work_amount = loop_info->get_work_amount();
    const auto work_amount_increment = loop_info->get_increment();

    const auto loop_bounds = loop_manager->get_loop_bounds(linear_ir, loop_id);

    std::vector<PortConnectorPtr> loop_end_inputs;
    loop_end_inputs.reserve(loop_entries.size() + loop_exits.size());
    loop_info->iterate_through_ports([&loop_end_inputs](const LoopPort& port) {
        loop_end_inputs.push_back(port.expr_port->get_port_connector_ptr());
    });

    const auto is_incremented = loop_info->get_is_incremented();
    const auto io_data_sizes = loop_info->get_data_sizes();

    // Should be inited by LoopInfo
    const auto is_dynamic_loop = is_loop_dynamic(loop_info);

    std::shared_ptr<op::LoopBegin> loop_begin = nullptr;
    std::shared_ptr<op::LoopEnd> loop_end = nullptr;
    if (is_dynamic_loop) {
        loop_begin = std::make_shared<op::LoopBeginDynamic>();
        loop_end = std::make_shared<op::LoopEndDynamic>(loop_begin, work_amount_increment, is_incremented, io_data_sizes,
                                                        loop_entries.size(), loop_exits.size(), loop_id);

    } else {
        const auto ptr_increments = loop_info->get_ptr_increments();
        const auto finalization_offsets = loop_info->get_finalization_offsets();

        loop_begin = std::make_shared<op::LoopBeginStatic>();
        loop_end = std::make_shared<op::LoopEndStatic>(loop_begin, work_amount, work_amount_increment, is_incremented, ptr_increments,
                                                       finalization_offsets, io_data_sizes, loop_entries.size(), loop_exits.size(), loop_id);
    }

    const auto outer_loop_ids = loop_manager->get_outer_expr_loops(*loop_bounds.first, loop_id);

    const auto loop_begin_expr = *linear_ir.insert_node(loop_begin, std::vector<PortConnectorPtr>{}, outer_loop_ids, false, loop_bounds.first);
    // Add LoopBegin port connector
    loop_end_inputs.push_back(loop_begin_expr->get_output_port_connector(0));
    linear_ir.insert_node(loop_end, loop_end_inputs, outer_loop_ids, false, loop_bounds.second);
}

bool InsertLoops::is_loop_dynamic(const UnifiedLoopInfoPtr& loop_info) {
    auto is_loop_port_dynamic = [](const UnifiedLoopInfo::LoopPortDesc& shifts) {
        return utils::is_dynamic_value(shifts.ptr_increment) || utils::is_dynamic_value(shifts.finalization_offset);
    };
    const auto& entry_shifts = loop_info->get_input_port_descs();
    const auto& exit_shifts = loop_info->get_output_port_descs();
    return utils::is_dynamic_value(loop_info->get_work_amount()) ||
           std::any_of(entry_shifts.cbegin(), entry_shifts.cend(), is_loop_port_dynamic) ||
           std::any_of(exit_shifts.cbegin(), exit_shifts.cend(), is_loop_port_dynamic);
}

bool InsertLoops::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertLoops")
    const auto& loop_manager = linear_ir.get_loop_manager();

    std::set<size_t> inserted_loops;
    for (auto expr_it = begin; expr_it != end; expr_it++) {
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
