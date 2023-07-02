// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/split_loops.hpp"

#include "snippets/lowered/pass/fuse_loops.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
using LoopManager = LinearIR::LoopManager;
using LoopInfoPtr = LoopManager::LoopInfoPtr;

SplitLoops::SplitLoops() : Pass() {}

bool SplitLoops::can_be_split(const LoopInfoPtr& current, const LoopInfoPtr& parent) {
    return current->work_amount == parent->work_amount && current->dim_idx == parent->dim_idx &&
           current->increment != parent->increment;
}

bool SplitLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SplitLoops")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    bool loop_was_split = false;
    for (const auto& expr : linear_ir) {
        const auto& loop_ids = expr->get_loop_ids();
        if (loop_ids.empty())
            continue;

        // Ticket: 113755
        // Note: we currently consider only the outermost loops for splitting
        // Splitting could also be done in a more general case, but the splitted loop and its parent must always
        // be in the same set of outer loops. Otherwise they won't be fused.
        const auto& loop_id = loop_ids.front();
        const auto loop = loop_manager->get_loop_info(loop_id);
        for (const auto& entry_point : loop->entry_points) {
            const auto& parent_port = entry_point.expr_port->get_port_connector_ptr()->get_source();
            const auto& parent_expr = parent_port.get_expr();
            const auto parent_loop_ids = parent_expr->get_loop_ids();
            if (parent_loop_ids.empty())
                continue;

            const auto& parent_loop_id = parent_loop_ids.front();
            const auto parent_loop_port = loop_manager->get_loop_port_by_expr_port(parent_port, parent_loop_id);
            // We don't split loop which are not compatible with parent loop because such loops will not be fused
            if (!FuseLoops::loop_ports_are_compatible(loop_manager, loop_id, parent_loop_id))
                continue;

            const auto parent_loop = loop_manager->get_loop_info(parent_loop_id);
            if (can_be_split(loop, parent_loop)) {
                loop_was_split = true;
                const bool split_parent = parent_loop->increment < loop->increment;
                const auto& loop_to_split = split_parent ? parent_loop : loop;
                const auto& loop_to_split_id = split_parent ? parent_loop_id : loop_id;
                const auto& loop_to_fuse = !split_parent ? parent_loop : loop;
                loop_to_split->work_amount = loop_to_fuse->increment;

                LinearIR::constExprIt loop_begin_pos, loop_end_pos;
                LoopManager::get_loop_bounds(linear_ir,
                                             loop_to_split->entry_points,
                                             loop_to_split->exit_points,
                                             loop_begin_pos,
                                             loop_end_pos,
                                             loop_to_split_id);
                const auto split_loop_id = loop_manager->mark_loop(loop_begin_pos,
                                                                   loop_end_pos,
                                                                   loop_to_fuse->work_amount,
                                                                   loop_to_fuse->increment,
                                                                   loop_to_split->dim_idx,
                                                                   loop_to_split->entry_points,
                                                                   loop_to_split->exit_points);
                loop_manager->get_loop_info(split_loop_id)->outer_splited_loop = true;
                break;
            }
        }
    }
    // Ticket: 113666
    // FuseLoops pass is explicitly run here in order to avoid unnecessary computations
    // in case if loops are not split but FuseLoops is registered in pass manager after SplitLoops
    if (loop_was_split)
        FuseLoops().run(linear_ir);
    return loop_was_split;
}
} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov