// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/fuse_loops.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
namespace {
bool is_loop_id_found(const std::vector<size_t>& ids, size_t id) {
    return std::find(ids.cbegin(), ids.cend(), id) != ids.cend();
}
} // namespace

FuseLoops::FuseLoops() : RangedPass() {}

bool FuseLoops::loop_ports_are_compatible(const LoopInfoPtr& loop_upper,
                                          const LoopInfoPtr& loop_lower) {
    auto found_port = [](const std::vector<LoopPort>& loop_ports, const ExpressionPort& target_port) {
        return std::find_if(loop_ports.cbegin(), loop_ports.cend(),
                            [&target_port](const LoopPort& loop_port) {return *(loop_port.expr_port.get()) == target_port; });
    };
    const auto& upper_exit_ports = loop_upper->get_output_ports();
    const auto& lower_entry_ports = loop_lower->get_input_ports();
    for (const auto& lower_entry_port : lower_entry_ports) {
        const auto& src_port = lower_entry_port.expr_port->get_port_connector_ptr()->get_source();
        const auto upper_exit_port_it = found_port(upper_exit_ports, src_port);
        if (upper_exit_port_it != upper_exit_ports.cend()) {
            const auto& upper_exit_port = *upper_exit_port_it;
            if (!lower_entry_port.is_incremented || !upper_exit_port.is_incremented)
                return false;
            if (lower_entry_port.dim_idx != upper_exit_port.dim_idx)
                return false;
        }
    }
    return true;
}

bool FuseLoops::can_be_fused(const UnifiedLoopInfoPtr& loop_upper, const UnifiedLoopInfoPtr& loop_lower) {
    OPENVINO_ASSERT(loop_upper != nullptr && loop_lower != nullptr, "LoopInfo is nullptr!");
    if (!loop_ports_are_compatible(loop_upper, loop_lower))
        return false;
    // Loop fusion is supported only if Loops have equal/broadcastable increments and work amounts.
    // Note: For example, Broadcastable work amounts are possible in the following case:
    //     Relu_0 [16x1]     Relu_1 [16x128]
    //                \           /
    //                 Add [16x128]
    // Because of expression order in linear IR and work of MarkLoop algorithm, there are 2 Inner Loops:
    //  - Relu_0 with work amount `1` and increment `1`
    //  - Relu_1 and Add with work amount `128` and increment `vector size`
    // We can fuse them into one Loop with work amount `128` and increment `vector size`
    const auto work_amount_upper = loop_upper->get_work_amount(), work_amount_lower = loop_lower->get_work_amount();
    const auto increment_upper = loop_upper->get_increment(), increment_lower = loop_lower->get_increment();
    const bool is_dynamic_case =
        (utils::is_dynamic_value(work_amount_upper) || utils::is_dynamic_value(work_amount_lower)) && increment_upper == increment_lower;
    const bool equal_parameters =
        (work_amount_upper == work_amount_lower) && increment_upper == increment_lower;
    const bool bcastable_upper = work_amount_upper == 1 && increment_upper == 1;
    const bool bcastable_lower = work_amount_lower == 1 && increment_lower == 1;
    // WA: we can't fuse 2 loops if one of them has first iteration handler but second hasn't,
    // because in this case Main/Tail body handlers of the loop wo first iter handler must be reset with new parameters
    // (e.g. tail size). This logic is not implemented for now, so fusion for such loops is skipped.
    const bool first_iter_handlers_match = loop_upper->get_handlers().get_passes<SpecificLoopIterType::FIRST_ITER>().empty() ==
                                           loop_lower->get_handlers().get_passes<SpecificLoopIterType::FIRST_ITER>().empty();
    // InnerSplittedUnifiedLoops can be fused only if they point to the same outer loop
    const auto& ispl_loop_upper = ov::as_type_ptr<InnerSplittedUnifiedLoopInfo>(loop_upper);
    const auto& ispl_loop_lower = ov::as_type_ptr<InnerSplittedUnifiedLoopInfo>(loop_lower);
    const auto& inner_splitted_loop_compatible =
        (!ispl_loop_upper && !ispl_loop_lower) ||
        (ispl_loop_upper && ispl_loop_lower && ispl_loop_upper->get_outer_splitted_loop_info() == ispl_loop_lower->get_outer_splitted_loop_info());
    return first_iter_handlers_match && inner_splitted_loop_compatible && (is_dynamic_case || equal_parameters || bcastable_upper || bcastable_lower);
}

void FuseLoops::move(LinearIR& linear_ir, const LoopManagerPtr& loop_manager, size_t loop_id,
                     LinearIR::constExprIt loop_begin_pos, LinearIR::constExprIt loop_end_pos, LinearIR::constExprIt pos) {
    // Inner Loops can contain ports which are ports of outer Loops as well.
    // When we move these inner loops, we can corrupt the sort of LoopPorts of outer Loops.
    // Firstly, we should get outer loop ids
    const auto outer_loop_ids = LoopManager::get_outer_expr_loops(*loop_begin_pos, loop_id);

     // Secondly, move expressions
    for (auto it = loop_begin_pos; it != loop_end_pos;) {
        auto expr_it = it;
        // After moving we will have `it` in new place in the current Loop,
        // but for markup we need have the expression from the target Loop.
        // Because of that we manually increment iterator before moving
        it = std::next(it);
        linear_ir.move(expr_it, pos);
    }
    // Thirdly, sort Loop Ports of outer Loops.
    loop_manager->sort_loop_ports(outer_loop_ids);
}

bool FuseLoops::fuse_upper_into_current(LinearIR& linear_ir, const LoopManagerPtr& loop_manager,
                                        const std::shared_ptr<ExpressionPort>& current_input_port,
                                        size_t current_loop_id, size_t target_loop_id,
                                        LinearIR::constExprIt& current_loop_begin_pos, LinearIR::constExprIt& current_loop_end_pos) {
    const auto& loop_current = loop_manager->get_loop_info<UnifiedLoopInfo>(current_loop_id);
    const auto& loop_target = loop_manager->get_loop_info<UnifiedLoopInfo>(target_loop_id);
    if (!can_be_fused(loop_target, loop_current))
        return false;

    // We can fuse Loop_up to Loop_down only in cases when other consumers of Loop_up are after Loop_down
    // Because Loop_up should be explicitly moved before Loop_down in linear IR, and we must save control dependency
    bool is_fusion_allowed = true;
    for (size_t i = 0; i < loop_target->get_output_ports().size() && is_fusion_allowed; ++i) {
        const auto target_output_port = loop_target->get_output_ports()[i];
        const auto consumer_inputs = target_output_port.expr_port->get_connected_ports();
        for (const auto& consumer_input : consumer_inputs) {
            const auto& consumer = consumer_input.get_expr();
            if (ov::is_type<ov::op::v0::Result>(consumer->get_node()) || consumer == current_input_port->get_expr())
                continue;
            // The fusing is only valid if target Loop consumer (the Consumer is outside of target Loop)
            // is after current Loop (after Loop_down).
            is_fusion_allowed = is_loop_id_found(consumer->get_loop_ids(), target_loop_id) || // is inside target Loop
                                is_loop_id_found(consumer->get_loop_ids(), current_loop_id) || // is inside current Loop
                                consumer->get_exec_num() > (*current_loop_end_pos)->get_exec_num(); // is after current Loop
        }
    }

    if (!is_fusion_allowed)
        return false;

    LinearIR::constExprIt target_loop_begin_pos, target_loop_end_pos;
    std::tie(target_loop_begin_pos, target_loop_end_pos) = loop_manager->get_loop_bounds(linear_ir, target_loop_id);
    loop_manager->fuse_loops(target_loop_begin_pos, target_loop_end_pos, target_loop_id, current_loop_id, false);
    const auto insertion_place = current_loop_begin_pos;
    const auto is_move_needed = target_loop_end_pos != current_loop_begin_pos;
    if (is_move_needed)
        move(linear_ir, loop_manager, current_loop_id, target_loop_begin_pos, target_loop_end_pos, insertion_place);

    // Update current Loop bounds:
    current_loop_begin_pos = target_loop_begin_pos;

    return true;
}

bool FuseLoops::fuse_lower_into_current(LinearIR& linear_ir, const LoopManagerPtr& loop_manager,
                                        const std::shared_ptr<ExpressionPort>& current_output_port,
                                        size_t current_loop_id, size_t target_loop_id,
                                        LinearIR::constExprIt& current_loop_begin_pos, LinearIR::constExprIt& current_loop_end_pos) {
    const auto& loop_current = loop_manager->get_loop_info<UnifiedLoopInfo>(current_loop_id);
    const auto& loop_target = loop_manager->get_loop_info<UnifiedLoopInfo>(target_loop_id);
    if (!can_be_fused(loop_current, loop_target))
        return false;

    // We can fuse Loop_down to Loop_up only in cases when other parents of Loop_down are before Loop_up
    // Because Loop_down should be explicitly moved after Loop_up in linear IR, and we must save control dependency
    bool is_fusion_allowed = true;
    for (size_t i = 0; i < loop_target->get_input_ports().size() && is_fusion_allowed; ++i) {
        const auto target_entry_port = loop_target->get_input_ports()[i];
        const auto parent_expr_output = *target_entry_port.expr_port->get_connected_ports().begin();
        const auto& parent_expr = parent_expr_output.get_expr();
        if (ov::is_type<ov::op::v0::Parameter>(parent_expr->get_node()) || parent_expr == current_output_port->get_expr())
            continue;
        is_fusion_allowed = is_loop_id_found(parent_expr->get_loop_ids(), current_loop_id) ||  // The parent expr is from the same current Loop
                            parent_expr->get_exec_num() < (*current_loop_begin_pos)->get_exec_num(); // The parent is before current Loop
    }

    if (!is_fusion_allowed)
        return false;

    LinearIR::constExprIt target_loop_begin_pos, target_loop_end_pos;
    std::tie(target_loop_begin_pos, target_loop_end_pos) = loop_manager->get_loop_bounds(linear_ir, target_loop_id);
    loop_manager->fuse_loops(target_loop_begin_pos, target_loop_end_pos, current_loop_id, target_loop_id);

    const auto insertion_place = current_loop_end_pos;
    const auto is_move_needed = insertion_place != target_loop_begin_pos;
    if (is_move_needed) {
        move(linear_ir, loop_manager, current_loop_id, target_loop_begin_pos, target_loop_end_pos, insertion_place);
    } else {
        current_loop_end_pos = target_loop_end_pos;
    }
    return true;
}

bool FuseLoops::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::FuseLoops")
    const auto& loop_manager = linear_ir.get_loop_manager();
    std::set<size_t> prev_fused_loops;

    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto expr = *expr_it;
        const auto& node = expr->get_node();
        if (ov::is_type<ov::op::v0::Parameter>(node) ||
            ov::is_type<ov::op::v0::Constant>(node) ||
            ov::is_type<ov::op::v0::Result>(node))
            continue;

        // Outer Loop ----> Inner Loop
        const auto& current_expr_loops = expr->get_loop_ids();
        const auto current_loop_depth = current_expr_loops.size();
        for (size_t i = 0; i < current_loop_depth; ++i) {
            const auto current_loop_id = current_expr_loops[i];
            // If the current Loop ID is in prev fused Loops, it means that on previous step all possible fusions are completed
            if (prev_fused_loops.count(current_loop_id) != 0)
                continue;

            // We fuse upper Loops into the current till we can do it.
            // After that we fuse lower Loops into the current till we can do it.
            // If we have fused on outputs we should verify possible fusions on inputs again because of new input ports
            bool need_fusion_checks = true;
            while (need_fusion_checks) {
                auto current_loop_info = loop_manager->get_loop_info(current_loop_id);
                LinearIR::constExprIt current_loop_begin_pos, current_loop_end_pos;
                std::tie(current_loop_begin_pos, current_loop_end_pos) = loop_manager->get_loop_bounds(linear_ir, current_loop_id);

                // Loop_0 (Upper)                 |
                //   |               =>           |
                // Loop_1 (Current)     Loop_0 + Loop_1 => new `Loop_1`
                // Make a copy of `input_ports` (not ref), since current_loop_info might be changed and ref will be invalid
                const auto input_ports = current_loop_info->get_input_ports();
                bool was_fusion_up = false;
                for (size_t in_port = 0; !was_fusion_up && in_port < input_ports.size(); ++in_port) {
                    const auto& input_port = input_ports[in_port];
                    const auto parent_expr_output = *input_port.expr_port->get_connected_ports().begin();
                    const auto& parent_expr = parent_expr_output.get_expr();
                    const auto parent = parent_expr->get_node();
                    if (ov::is_type<ov::op::v0::Constant>(parent) ||
                        ov::is_type<ov::op::v0::Parameter>(parent) ||
                        ov::is_type<op::Buffer>(parent)) {
                        continue;
                    }

                    const auto& upper_loop_ids = parent_expr->get_loop_ids();
                    if (upper_loop_ids.empty())
                        continue;

                    size_t loop_idx = 0;
                    bool allowed = true;
                    for (; loop_idx < std::min(i, upper_loop_ids.size()) && allowed; ++loop_idx) {
                        if (current_expr_loops[loop_idx] != upper_loop_ids[loop_idx])
                            allowed = false;
                    }
                    // There are different outer loops or all Loop IDs are the same
                    if (!allowed || loop_idx == upper_loop_ids.size())
                        continue;

                    const auto upper_loop_id = upper_loop_ids[loop_idx];
                    OPENVINO_ASSERT(current_loop_id != upper_loop_id,
                                    "Loops cannot have parents of input ports with the same identifier (", upper_loop_id, ")");
                    if (fuse_upper_into_current(linear_ir, loop_manager, input_port.expr_port, current_loop_id, upper_loop_id,
                                                current_loop_begin_pos, current_loop_end_pos)) {
                        was_fusion_up = true;
                        prev_fused_loops.insert(current_loop_id);
                        current_loop_info = loop_manager->get_loop_info(current_loop_id);
                    }
                }

                // If Loops were fused and there are new input_ports, we should check for possible fusion again
                if (was_fusion_up && input_ports != current_loop_info->get_input_ports())
                    continue;

                // Loop_0 (Current)    Loop_0 + Loop_1 => new `Loop_0`
                //   |               =>           |
                // Loop_1 (Lower)                 |
                bool was_fusion_down = false;
                const auto& output_ports = current_loop_info->get_output_ports();
                for (size_t out_port = 0; !was_fusion_down && out_port < output_ports.size(); ++out_port) {
                    const auto& output_port = output_ports[out_port];
                    const auto consumer_exprs_inputs = output_port.expr_port->get_connected_ports();
                    for (const auto& consumer_expr_input : consumer_exprs_inputs) {
                        const auto& consumer_expr = consumer_expr_input.get_expr();
                        const auto consumer = consumer_expr->get_node();
                        if (ov::is_type<ov::op::v0::Result>(consumer) ||
                            ov::is_type<op::Buffer>(consumer)) {
                            continue;
                        }

                        const auto& lower_loop_ids = consumer_expr->get_loop_ids();
                        if (lower_loop_ids.empty())
                            continue;

                        size_t loop_idx = 0;
                        bool allowed = true;
                        for (; loop_idx < std::min(i, lower_loop_ids.size()) && allowed; ++loop_idx) {
                            if (current_expr_loops[loop_idx] != lower_loop_ids[loop_idx])
                                allowed = false;
                        }
                        // There are different outer loops or all Loop IDs are the same
                        if (!allowed || loop_idx == lower_loop_ids.size())
                            continue;

                        const auto lower_loop_id = lower_loop_ids[loop_idx];
                        // if expressions have the same loop ID, it means that they're from the same Loop (one of consumers is in Loop)
                        if (current_loop_id == lower_loop_id)
                            continue;

                        if (fuse_lower_into_current(linear_ir, loop_manager, output_port.expr_port, current_loop_id, lower_loop_id,
                                                    current_loop_begin_pos, current_loop_end_pos)) {
                            was_fusion_down = true;
                            prev_fused_loops.insert(current_loop_id);
                            current_loop_info = loop_manager->get_loop_info(current_loop_id);
                            // Need to check for possible fusion again because of new input expressions for Loop
                            break;
                        }
                    }
                }

                // We iterated by each output port and didn't fuse new Loops -> we can finish check for possible fusions on outputs.
                if (!was_fusion_down)
                    need_fusion_checks = false;
            }
        }
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
