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
using LoopInfo = LoopManager::LoopInfo;
using LoopInfoPtr = LoopManager::LoopInfoPtr;

SplitLoops::SplitLoops() : Pass() {}

bool SplitLoops::can_be_split(const LoopInfoPtr& current, const LoopInfoPtr& parent) {
    const auto current_dim_idx = current->get_dim_idx();
    const auto parent_dim_idx = parent->get_dim_idx();
    const bool equal_dim_idxes = current_dim_idx != LoopInfo::UNDEFINED_DIM_IDX && current_dim_idx == parent_dim_idx;
    return current->get_work_amount() == parent->get_work_amount() && current->get_increment() != parent->get_increment() && equal_dim_idxes;
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
        for (const auto& entry_point : loop->get_entry_points()) {
            const auto& parent_port = entry_point.expr_port->get_port_connector_ptr()->get_source();
            const auto& parent_expr = parent_port.get_expr();
            const auto& parent_loop_ids = parent_expr->get_loop_ids();
            if (parent_loop_ids.empty())
                continue;

            const auto& parent_loop_id = parent_loop_ids.front();
            const auto parent_loop = loop_manager->get_loop_info(parent_loop_id);

            const bool split_parent = parent_loop->get_increment() < loop->get_increment();
            const auto upper_loop = std::make_shared<LoopManager::LoopInfo>(*parent_loop);
            const auto lower_loop = std::make_shared<LoopManager::LoopInfo>(*loop);
            if (split_parent)
                upper_loop->set_increment(loop->get_increment());
            else
                lower_loop->set_increment(parent_loop->get_increment());
            // We don't split loop which are not compatible with parent loop because such loops will not be fused
            if (!FuseLoops::can_be_fused(upper_loop, lower_loop))
                continue;

            if (can_be_split(loop, parent_loop)) {
                loop_was_split = true;
                const auto& loop_to_split = split_parent ? parent_loop : loop;
                const auto& loop_to_split_id = split_parent ? parent_loop_id : loop_id;
                const auto& loop_to_fuse = !split_parent ? parent_loop : loop;
                loop_to_split->set_work_amount(loop_to_fuse->get_increment());

                const auto loop_bounds = LoopManager::get_loop_bounds(linear_ir, loop_to_split_id,
                                                                      loop_to_split->get_entry_points(),
                                                                      loop_to_split->get_exit_points());
                const auto split_loop_id = loop_manager->mark_loop(loop_bounds.first,
                                                                   loop_bounds.second,
                                                                   loop_to_fuse->get_work_amount(),
                                                                   loop_to_fuse->get_increment(),
                                                                   loop_to_split->get_dim_idx(),
                                                                   loop_to_split->get_entry_points(),
                                                                   loop_to_split->get_exit_points());
                loop_manager->get_loop_info(split_loop_id)->set_outer_splited_loop(true);
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