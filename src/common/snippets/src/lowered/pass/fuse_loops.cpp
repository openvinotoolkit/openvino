// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/fuse_loops.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
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

using LoopManager = LinearIR::LoopManager;
using LoopInfoPtr = LoopManager::LoopInfoPtr;

FuseLoops::FuseLoops() : Pass() {}

bool FuseLoops::can_be_fused(const LoopInfoPtr& loop_current, const LoopInfoPtr& loop_target) {
    auto current_work_amount = loop_current->work_amount;
    auto target_work_amount = loop_target->work_amount;
    // Loop fusion is supported only if Loops have equal increments and the equal/broadcastable work amounts.
    // Note: For example, Broadcastable work amounts are possible in the following case:
    //     Relu_0 [16x1]     Relu_1 [16x128]
    //                \           /
    //                 Add [16x128]
    // Because of expression order in linear IR and work of MarkLoop algorithm, there are 2 Inner Loops:
    //  - Relu_0 with work amount `1` and increment `vector size`
    //  - Relu_1 and Add with work amount `128` and increment `vector size`
    // We can fuse them into one Loop with work amount `128` and increment `vector size`
    const auto supported_work_amount = current_work_amount == target_work_amount || current_work_amount == 1 || target_work_amount == 1;
    const auto supported_increment = loop_current->increment == loop_target->increment;
    const auto supported_dim_idxs = loop_current->dim_idx == loop_target->dim_idx;
    return supported_work_amount && supported_increment && supported_dim_idxs;
}

void FuseLoops::fuse_points(std::vector<LinearIR::LoopManager::LoopPort>& exit_points,
                            std::vector<LinearIR::LoopManager::LoopPort>& entry_points,
                            LinearIR::constExprIt loop_begin_pos, LinearIR::constExprIt loop_end_pos) {
    std::vector<LinearIR::LoopManager::LoopPort> new_exit_points;
    for (const auto& exit_point : exit_points) {
        const auto consumers_inputs = exit_point.port->get_connected_ports();

        std::set<LinearIR::LoopManager::LoopPort> mapped_entry_points;
        std::set<ExpressionPtr> outside_consumers;
        for (const auto& consumer_input : consumers_inputs) {
            const auto entry_point_it = std::find_if(entry_points.begin(), entry_points.end(),
                                                     [&consumer_input](const LoopManager::LoopPort& point) { return *point.port.get() == consumer_input; });
            if (entry_point_it != entry_points.end()) {
                mapped_entry_points.insert(*entry_point_it);
                continue;
            }

            const auto& consumer = consumer_input.get_expr();
            const auto inside_it = std::find(loop_begin_pos, loop_end_pos, consumer);
            if (inside_it == loop_end_pos) {
                outside_consumers.insert(consumer);
            }
        }

        // Remove entry points which are mapped
        auto last_point = entry_points.end();
        for (const auto& mapped_entry_point : mapped_entry_points) {
            last_point = std::remove(entry_points.begin(), last_point, mapped_entry_point);
        }
        entry_points.resize(entry_points.size() - mapped_entry_points.size());

        // Leave exit point if there are consumers outside after fusion
        if (!outside_consumers.empty()) {
            new_exit_points.push_back(exit_point);
        }
    }

    exit_points = new_exit_points;
}

bool FuseLoops::fuse_upper_into_current(LinearIR& linear_ir, const LinearIR::LoopManagerPtr& loop_manager,
                                        const std::shared_ptr<ExpressionPort>& current_entry_point,
                                        size_t current_loop_id, size_t target_loop_id,
                                        LinearIR::constExprIt& current_loop_begin_pos, LinearIR::constExprIt& current_loop_end_pos) {
    const auto& loop_current = loop_manager->get_loop_info(current_loop_id);
    const auto& loop_target = loop_manager->get_loop_info(target_loop_id);
    if (!can_be_fused(loop_current, loop_target))
        return false;

    LinearIR::constExprIt target_loop_begin_pos, target_loop_end_pos;
    loop_manager->get_loop_bounds(linear_ir, target_loop_id, target_loop_begin_pos, target_loop_end_pos);

    // We can fuse Loop_up to Loop_down only in cases when other consumers of Loop_up are after Loop_down
    // Because Loop_up should be explicitly moved before Loop_down in linear IR, and we must save control dependency
    bool is_fusion_allowed = true;
    for (size_t i = 0; i < loop_target->exit_points.size() && is_fusion_allowed; ++i) {
        const auto target_exit_point = loop_target->exit_points[i];
        const auto consumer_inputs = target_exit_point.port->get_connected_ports();
        for (const auto& consumer_input : consumer_inputs) {
            const auto& consumer = consumer_input.get_expr();
            if (ov::is_type<ov::op::v0::Result>(consumer->get_node()) || consumer == current_entry_point->get_expr())
                continue;
            // The fusing is only valid if target Loop consumer (the Consumer is outside of target Loop)
            // is after current Loop (after Loop_down).
            is_fusion_allowed = is_loop_id_found(consumer->get_loop_ids(), target_loop_id) || // is inside target Loop
                                is_loop_id_found(consumer->get_loop_ids(), current_loop_id) || // is inside current Loop
                                std::find(current_loop_end_pos, linear_ir.cend(), consumer) != linear_ir.end();  // is after current Loop
        }
    }

    if (!is_fusion_allowed)
        return false;

    // Update entry and exit points in current Loop information before moving till Loop iterators are valid
    auto current_entry_points = loop_current->entry_points;
    auto current_exit_points = loop_current->exit_points;
    auto target_entry_points = loop_target->entry_points;
    auto target_exit_points = loop_target->exit_points;
    fuse_points(target_exit_points, current_entry_points, target_loop_begin_pos, target_loop_end_pos);

    const auto insertion_place = current_loop_begin_pos;
    const auto is_move_needed = target_loop_end_pos != current_loop_begin_pos;
    for (auto it = target_loop_begin_pos; it != target_loop_end_pos;) {
        auto expr_it = it;
        const auto& expr = *expr_it;
        // After moving we will have `it` in new place in the current Loop,
        // but for markup we need have the expression from the target Loop.
        // Because of that we manually increment iterator before moving
        it = std::next(it);
        loop_manager->replace_loop_id(expr, target_loop_id, current_loop_id);
        if (is_move_needed)
            linear_ir.move(expr_it, insertion_place);
    }

    // Update current Loop bounds:
    current_loop_begin_pos = target_loop_begin_pos;

    // Update work_amount for Loop (increment is constant because increments must be the identical for fusion):
    loop_current->work_amount = std::max(loop_current->work_amount, loop_target->work_amount);

    std::vector<LoopManager::LoopPort> new_entries = target_entry_points;
    new_entries.insert(new_entries.end(), current_entry_points.begin(), current_entry_points.end());
    std::vector<LoopManager::LoopPort> new_exits = target_exit_points;
    new_exits.insert(new_exits.end(), current_exit_points.begin(), current_exit_points.end());

    loop_current->entry_points = new_entries;
    loop_current->exit_points = new_exits;

    return true;
}

bool FuseLoops::fuse_lower_into_current(LinearIR& linear_ir, const LinearIR::LoopManagerPtr& loop_manager,
                                        const std::shared_ptr<ExpressionPort>& current_exit_point,
                                        size_t current_loop_id, size_t target_loop_id,
                                        LinearIR::constExprIt& current_loop_begin_pos, LinearIR::constExprIt& current_loop_end_pos) {
    const auto& loop_current = loop_manager->get_loop_info(current_loop_id);
    const auto& loop_target = loop_manager->get_loop_info(target_loop_id);
    if (!can_be_fused(loop_current, loop_target))
        return false;

    // We can fuse Loop_down to Loop_up only in cases when other parents of Loop_down are before Loop_up
    // Because Loop_down should be explicitly moved after Loop_up in linear IR, and we must save control dependency
    bool is_fusion_allowed = true;
    for (size_t i = 0; i < loop_target->entry_points.size() && is_fusion_allowed; ++i) {
        const auto target_entry_port = loop_target->entry_points[i];
        const auto parent_expr_output = *target_entry_port.port->get_connected_ports().begin();
        const auto& parent_expr = parent_expr_output.get_expr();
        if (ov::is_type<ov::op::v0::Parameter>(parent_expr->get_node()) || parent_expr == current_exit_point->get_expr())
            continue;
        is_fusion_allowed = is_loop_id_found(parent_expr->get_loop_ids(), current_loop_id) ||  // The parent expr is from the same current Loop
                            std::find(linear_ir.cbegin(), current_loop_begin_pos, parent_expr) != current_loop_begin_pos; // The parent is before current Loop
    }

    if (!is_fusion_allowed)
        return false;

    LinearIR::constExprIt target_loop_begin_pos, target_loop_end_pos;
    loop_manager->get_loop_bounds(linear_ir, target_loop_id, target_loop_begin_pos, target_loop_end_pos);

    // Update entry and exit points in current Loop information before moving till Loop iterators are valid
    auto current_entry_points = loop_current->entry_points;
    auto current_exit_points = loop_current->exit_points;
    auto target_entry_points = loop_target->entry_points;
    auto target_exit_points = loop_target->exit_points;
    fuse_points(current_exit_points, target_entry_points, current_loop_begin_pos, current_loop_end_pos);

    const auto insertion_place = current_loop_end_pos;
    const auto is_move_needed = insertion_place != target_loop_begin_pos;
    for (auto it = target_loop_begin_pos; it != target_loop_end_pos;) {
        auto expr_it = it;
        const auto& expr = *expr_it;
        // After moving we will have `it` in new place in the current Loop,
        // but for markup we need have the expression from the target Loop.
        // Because of that we manually increment iterator before moving
        it = std::next(it);
        loop_manager->replace_loop_id(expr, target_loop_id, current_loop_id);
        if (is_move_needed)
            linear_ir.move(expr_it, insertion_place);
    }

    // Update current Loop bounds:
    if (!is_move_needed)
        current_loop_end_pos = target_loop_end_pos;

    // Update work_amount for Loop (increment is constant because increments must be the identical for fusion):
    loop_current->work_amount = std::max(loop_current->work_amount, loop_target->work_amount);

    std::vector<LoopManager::LoopPort>& new_entries = current_entry_points;
    new_entries.insert(new_entries.end(), target_entry_points.begin(), target_entry_points.end());
    std::vector<LoopManager::LoopPort>& new_exits = current_exit_points;
    new_exits.insert(new_exits.end(), target_exit_points.begin(), target_exit_points.end());

    loop_current->entry_points = new_entries;
    loop_current->exit_points = new_exits;

    return true;
}

bool FuseLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::FuseLoops")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    std::set<size_t> prev_fused_loops;

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto expr = *expr_it;
        const auto& node = expr->get_node();
        if (ov::is_type<ov::op::v0::Parameter>(node) ||
            ov::is_type<ov::op::v0::Constant>(node) ||
            ov::is_type<ov::op::v0::Result>(node))
            continue;

        // Outer Loop ----> Inner Loop
        const auto current_expr_loops = expr->get_loop_ids();
        const auto current_loop_depth = current_expr_loops.size();
        for (size_t i = 0; i < current_loop_depth; ++i) {
            const auto current_loop_id = current_expr_loops[i];
            // If the current Loop ID is in prev fused Loops, it means that on previous step all possible fusions are completed
            if (prev_fused_loops.count(current_loop_id) != 0)
                continue;

            const auto current_loop_info = loop_manager->get_loop_info(current_loop_id);
            LinearIR::constExprIt current_loop_begin_pos, current_loop_end_pos;
            loop_manager->get_loop_bounds(linear_ir, current_loop_id, current_loop_begin_pos, current_loop_end_pos);

            // We fuse upper Loops into the current till we can do it.
            // After that we fuse lower Loops into the current till we can do it.
            // If we have fused on outputs we should verify possible fusions on inputs again because of new entry points
            bool need_fusion_checks = true;
            while (need_fusion_checks) {
                // Loop_0 (Upper)                 |
                //   |               =>           |
                // Loop_1 (Current)     Loop_0 + Loop_1 => new `Loop_1`
                auto entry_points = current_loop_info->entry_points;
                bool was_fusion_up = false;
                for (size_t in_port = 0; in_port < entry_points.size() && !was_fusion_up; ++in_port) {
                    const auto entry_point = entry_points[in_port];
                    const auto parent_expr_output = *entry_point.port->get_connected_ports().begin();
                    const auto& parent_expr = parent_expr_output.get_expr();
                    const auto parent = parent_expr->get_node();
                    if (ov::is_type<ov::op::v0::Constant>(parent) ||
                        ov::is_type<ov::op::v0::Parameter>(parent) ||
                        ov::is_type<op::Buffer>(parent)) {
                        continue;
                    }

                    const auto upper_loop_ids = parent_expr->get_loop_ids();
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
                                    "Loops cannot have parents of entry points with the same identifier");
                    if (fuse_upper_into_current(linear_ir, loop_manager, entry_point.port, current_loop_id, upper_loop_id,
                                                current_loop_begin_pos, current_loop_end_pos)) {
                        was_fusion_up = true;
                        loop_manager->remove_loop_info(upper_loop_id);
                        prev_fused_loops.insert(current_loop_id);
                    }
                }

                // If Loops were fused and there are new entry_points, we should check for possible fusion again
                if (was_fusion_up && entry_points != current_loop_info->entry_points)
                    continue;

                // Loop_0 (Current)    Loop_0 + Loop_1 => new `Loop_0`
                //   |               =>           |
                // Loop_1 (Lower)                 |
                auto exit_points = current_loop_info->exit_points;
                bool was_fusion_down = false;
                for (size_t out_port = 0; out_port < exit_points.size() && !was_fusion_down; ++out_port) {
                    const auto exit_point = exit_points[out_port];
                    const auto consumer_exprs_inputs = exit_point.port->get_connected_ports();
                    for (const auto& consumer_expr_input : consumer_exprs_inputs) {
                        const auto& consumer_expr = consumer_expr_input.get_expr();
                        const auto consumer = consumer_expr->get_node();
                        if (ov::is_type<ov::op::v0::Result>(consumer) ||
                            ov::is_type<op::Buffer>(consumer)) {
                            continue;
                        }

                        const auto lower_loop_ids = consumer_expr->get_loop_ids();
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

                        if (fuse_lower_into_current(linear_ir, loop_manager, exit_point.port, current_loop_id, lower_loop_id,
                                                    current_loop_begin_pos, current_loop_end_pos)) {
                            was_fusion_down = true;
                            loop_manager->remove_loop_info(lower_loop_id);
                            prev_fused_loops.insert(current_loop_id);
                            // Need to check for possible fusion again because of new input expressions for Loop
                            break;
                        }
                    }
                }

                // We iterated by each exit point and didn't fuse new Loops -> we can finish check for possible fusions on outputs.
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
