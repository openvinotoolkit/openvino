// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/split_loops.hpp"

#include "snippets/lowered/pass/fuse_loops.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/iter_handler.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

SplitLoops::SplitLoops() : RangedPass() {}

bool SplitLoops::can_be_split(const UnifiedLoopInfoPtr& loop_to_split, const UnifiedLoopInfoPtr& loop_to_fuse) {
    OPENVINO_ASSERT(loop_to_split != nullptr && loop_to_fuse != nullptr, "LoopInfo is nullptr!");
    const auto current_dim_idx = loop_to_split->get_dim_idx();
    const auto parent_dim_idx = loop_to_fuse->get_dim_idx();
    const auto& handlers = loop_to_split->get_handlers();
    const bool equal_dim_idxes = current_dim_idx != LoopInfo::UNDEFINED_DIM_IDX && current_dim_idx == parent_dim_idx;
    const bool only_main_body = handlers.get_passes<SpecificLoopIterType::FIRST_ITER>().empty() &&
                                handlers.get_passes<SpecificLoopIterType::LAST_ITER>().empty();
    // TODO [141735] : At the moment Splitted loops are not supported in dynamic case
    const auto are_static = !loop_to_split->is_dynamic() && !loop_to_fuse->is_dynamic();
    return are_static && loop_to_split->get_work_amount() == loop_to_fuse->get_work_amount() &&
           loop_to_split->get_increment() != loop_to_fuse->get_increment() && equal_dim_idxes && only_main_body;
}

bool SplitLoops::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SplitLoops")
    const auto& loop_manager = linear_ir.get_loop_manager();
    bool loop_was_split = false;
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = *expr_it;
        const auto& loop_ids = expr->get_loop_ids();
        if (loop_ids.empty())
            continue;

        // Ticket: 113755
        // Note: we currently consider only the outermost loops for splitting
        // Splitting could also be done in a more general case, but the splitted loop and its parent must always
        // be in the same set of outer loops. Otherwise they won't be fused.
        const auto& loop_id = loop_ids.front();
        const auto loop = loop_manager->get_loop_info<UnifiedLoopInfo>(loop_id);
        for (const auto& input_port : loop->get_input_ports()) {
            const auto& parent_port = input_port.expr_port->get_port_connector_ptr()->get_source();
            const auto& parent_expr = parent_port.get_expr();
            const auto& parent_loop_ids = parent_expr->get_loop_ids();
            if (parent_loop_ids.empty())
                continue;

            const auto& parent_loop_id = parent_loop_ids.front();
            const auto parent_loop = loop_manager->get_loop_info<UnifiedLoopInfo>(parent_loop_id);

            const bool split_parent = parent_loop->get_increment() < loop->get_increment();
            const auto upper_loop = std::make_shared<UnifiedLoopInfo>(*parent_loop);
            const auto lower_loop = std::make_shared<UnifiedLoopInfo>(*loop);
            if (split_parent)
                upper_loop->set_increment(loop->get_increment());
            else
                lower_loop->set_increment(parent_loop->get_increment());

            const auto& loop_to_split = split_parent ? parent_loop : loop;
            const auto& loop_to_fuse = !split_parent ? parent_loop : loop;
            // We don't split loop which are not compatible with parent loop because such loops will not be fused
            if (FuseLoops::can_be_fused(upper_loop, lower_loop) && can_be_split(loop_to_split, loop_to_fuse)) {
                loop_was_split = true;
                loop_to_split->set_work_amount(loop_to_fuse->get_increment());
                // Since the loop has work amount equal to increment of outer loop, not broadcasted dimension,
                // we should set `work_amount_const = true` to avoid rewriting in common loop intiialization passes (for example, `InitLoops`)
                loop_to_split->set_work_amount_const(true);

                const auto& loop_to_split_id = split_parent ? parent_loop_id : loop_id;
                const auto loop_bounds = LoopManager::get_loop_bounds(linear_ir, loop_to_split_id,
                                                                      loop_to_split->get_input_ports(),
                                                                      loop_to_split->get_output_ports());
                const auto split_loop_id = loop_manager->mark_loop(loop_bounds.first,
                                                                   loop_bounds.second,
                                                                   loop_to_fuse->get_work_amount(),
                                                                   loop_to_fuse->get_increment(),
                                                                   loop_to_split->get_dim_idx(),
                                                                   loop_to_split->get_input_ports(),
                                                                   loop_to_split->get_output_ports());
                const auto& new_loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(split_loop_id);
                const auto work_amount = loop_to_fuse->get_work_amount();
                const auto increment = loop_to_fuse->get_increment();
                const auto tail_size = work_amount % increment;
                auto new_handlers = loop_to_split->get_handlers();
                if (tail_size != 0) {
                    new_handlers.register_pass<SpecificLoopIterType::LAST_ITER, TransformInnerSplitLoop>(tail_size);
                }
                new_loop_info->set_handlers(new_handlers);
                break;
            }
        }
    }
    // Ticket: 113666
    // FuseLoops pass is explicitly run here in order to avoid unnecessary computations
    // in case if loops are not split but FuseLoops is registered in pass manager after SplitLoops
    if (loop_was_split)
        FuseLoops().run(linear_ir, begin, end);
    return loop_was_split;
}
} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov