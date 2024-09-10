// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/split_loops.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/fuse_loops.hpp"
#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/lowered/pass/iter_handler.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/loop_utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool SplitLoops::can_be_split(const UnifiedLoopInfoPtr& loop_to_split, const UnifiedLoopInfoPtr& loop_to_fuse) {
    OPENVINO_ASSERT(loop_to_split != nullptr && loop_to_fuse != nullptr, "LoopInfo is nullptr!");
    const auto current_dim_idx = loop_to_split->get_dim_idx();
    const auto parent_dim_idx = loop_to_fuse->get_dim_idx();
    const auto& handlers = loop_to_split->get_handlers();
    const bool equal_dim_idxes = current_dim_idx != LoopInfo::UNDEFINED_DIM_IDX && current_dim_idx == parent_dim_idx;
    const bool only_main_body = handlers.get_passes<SpecificLoopIterType::FIRST_ITER>().empty() &&
                                handlers.get_passes<SpecificLoopIterType::LAST_ITER>().empty();
    return loop_to_split->get_work_amount() == loop_to_fuse->get_work_amount() &&
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
                split(linear_ir, split_parent ? parent_loop_id : loop_id, loop_to_fuse->get_increment());
                loop_was_split = true;
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

void SplitLoops::split(LinearIR& linear_ir, size_t loop_to_split_id, size_t outer_increment) {
    const auto& loop_manager = linear_ir.get_loop_manager();

    const auto& inner_loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(loop_to_split_id);
    const auto loop_bounds = LoopManager::get_loop_bounds(linear_ir, loop_to_split_id,
                                                          inner_loop_info->get_input_ports(),
                                                          inner_loop_info->get_output_ports());
    const auto outer_loop_id = loop_manager->mark_loop(loop_bounds.first, loop_bounds.second, inner_loop_info->get_work_amount(),
                                                       outer_increment, inner_loop_info->get_dim_idx(),
                                                       inner_loop_info->get_input_ports(), inner_loop_info->get_output_ports(), false);
    const auto& outer_loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(outer_loop_id);

    const auto& inner_splitted_loop_info =
        std::make_shared<InnerSplittedUnifiedLoopInfo>(inner_loop_info->get_increment(), inner_loop_info->get_input_ports(),
                                                       inner_loop_info->get_output_ports(), inner_loop_info->get_input_port_descs(),
                                                       inner_loop_info->get_output_port_descs(), inner_loop_info->get_handlers(),
                                                       outer_loop_info);
    loop_manager->replace_with_new_loop(linear_ir, loop_bounds.first, loop_bounds.second, inner_splitted_loop_info, loop_to_split_id);

    if (!outer_loop_info->get_handlers().get_passes<SpecificLoopIterType::FIRST_ITER>().empty()) {
        outer_loop_info->register_pass_to_handler<SpecificLoopIterType::FIRST_ITER, TransformInnerSplitLoop>();
    }
    outer_loop_info->register_pass_to_handler<SpecificLoopIterType::MAIN_BODY, TransformInnerSplitLoop>();
    outer_loop_info->register_pass_to_handler<SpecificLoopIterType::LAST_ITER, TransformInnerSplitLoop>();
}

namespace {
InnerSplittedUnifiedLoopInfoPtr make_own_inner_splitted_unified_loop_info(const ExpandedLoopInfoPtr& inner_expanded,
                                                                          const ExpandedLoopInfoPtr& outer_expanded,
                                                                          const InnerSplittedUnifiedLoopInfoPtr& existing_inner_unified) {
    const auto loop_info =
        std::make_shared<InnerSplittedUnifiedLoopInfo>(inner_expanded->get_increment(), inner_expanded->get_input_ports(),
                                                       inner_expanded->get_output_ports(), existing_inner_unified->get_input_port_descs(),
                                                       existing_inner_unified->get_output_port_descs(), existing_inner_unified->get_handlers(),
                                                       outer_expanded);
    ov::snippets::utils::update_runtime_parameters(loop_info);
    return loop_info;
}
ExpandedLoopInfoPtr make_own_inner_splitted_expanded_loop_info(const ExpandedLoopInfoPtr& inner_expanded,
                                                               const InnerSplittedUnifiedLoopInfoPtr& inner_unified) {
    return std::make_shared<ExpandedLoopInfo>(inner_unified->get_work_amount(), inner_unified->get_increment(),
                                              inner_unified->get_input_ports(), inner_unified->get_output_ports(),
                                              inner_unified->get_ptr_increments(),
                                              inner_unified->get_finalization_offsets(),
                                              inner_unified->get_data_sizes(), inner_expanded->get_type(),
                                              inner_unified, inner_expanded->is_evaluate_once());
}
}  // namespace

bool SplitLoops::TransformInnerSplitLoop::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    OPENVINO_ASSERT(end != linear_ir.cend(), "Incorrect LinearIR range for processing");
    const auto& expr = *end;
    const auto node = expr->get_node();
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(node);
    OPENVINO_ASSERT(loop_end, "the last operation in range must be LoopEnd");

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& outer_loop_info = loop_manager->get_loop_info<ExpandedLoopInfo>(loop_end->get_id());
    const auto current_dim_idx = outer_loop_info->get_dim_idx();
    OPENVINO_ASSERT(current_dim_idx != LoopInfo::UNDEFINED_DIM_IDX,
                    "Outer splitted loop unexpectedly iterates by several dimension indices");

    bool modified = false;
    for (auto it = begin; it != end; ++it) {
        const auto& expr = *it;
        const auto inner_loop_end = ov::as_type_ptr<op::LoopEnd>(expr->get_node());
        if (!inner_loop_end)
            continue;

        // There is already ExpandedLoopInfo
        const auto& inner_expanded_loop_info = loop_manager->get_loop_info<ExpandedLoopInfo>(inner_loop_end->get_id());
        const auto inner_unified_loop_info = ov::as_type_ptr<InnerSplittedUnifiedLoopInfo>(inner_expanded_loop_info->get_unified_loop_info());
        if (!inner_unified_loop_info || inner_unified_loop_info->get_outer_splitted_loop_info() != outer_loop_info->get_unified_loop_info())
            continue;

        OPENVINO_ASSERT(current_dim_idx == inner_unified_loop_info->get_dim_idx(), "Incorrect processing dim index of splitted loops");
        OPENVINO_ASSERT(inner_expanded_loop_info->get_type() == SpecificLoopIterType::MAIN_BODY, "InnerSplittedLoop must be Main Body of loop");

        // We have to make a new UnifiedLoopInfo to distinguish it from other unified loops in other specific iterations of outer loop.
        const auto inner_splitted_unified_loop_info = make_own_inner_splitted_unified_loop_info(inner_expanded_loop_info, outer_loop_info,
                                                                                                inner_unified_loop_info);

        // We have to replace existing ExpandedLoopInfo with new one to have the own InnerSplittedUnifiedLoopInfo and
        // distinguish it from other expanded loops in other specific iterations of outer loop.
        const auto new_expanded_inner_loop_info = make_own_inner_splitted_expanded_loop_info(inner_expanded_loop_info, inner_splitted_unified_loop_info);
        const auto inner_begin = linear_ir.find_before(it, linear_ir.get_expr_by_node(inner_loop_end->get_loop_begin()));
        const auto new_id = loop_manager->replace_with_new_loop(linear_ir, inner_begin, std::next(it), new_expanded_inner_loop_info, inner_loop_end->get_id());

        // [147894] : Update inner LoopEnd expression
        inner_loop_end->set_id(new_id);
        inner_loop_end->set_work_amount(new_expanded_inner_loop_info->get_work_amount());
        inner_loop_end->set_increment(new_expanded_inner_loop_info->get_increment());
        inner_loop_end->set_finalization_offsets(new_expanded_inner_loop_info->get_finalization_offsets());
    }
    return modified;
}

std::shared_ptr<pass::PassBase> SplitLoops::TransformInnerSplitLoop::merge(const std::shared_ptr<pass::PassBase>& other) {
    return !other || ov::is_type<TransformInnerSplitLoop>(other) ? std::make_shared<TransformInnerSplitLoop>() : nullptr;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov