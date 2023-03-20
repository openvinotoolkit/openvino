// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/load_store_insertion.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

using LoweredLoopManager = LoweredExprIR::LoweredLoopManager;
using LoweredLoopInfoPtr = LoweredLoopManager::LoweredLoopInfoPtr;

LoadStoreInsertion::LoadStoreInsertion(size_t vector_size) : m_vector_size(vector_size) {}

void LoadStoreInsertion::update_loops(const LoweredExprIR::LoweredLoopManagerPtr& loop_manager, const std::vector<size_t>& loop_identifies,
                                      const LoweredExprPort& actual_port, const std::vector<LoweredExprPort>& target_ports, bool is_entry) {
    for (auto loop_id : loop_identifies) {
        if (loop_id != LoweredExpr::LOOP_NULL_ID)
            update_loop(loop_manager->get(loop_id), actual_port, target_ports, is_entry);
    }
}

void LoadStoreInsertion::update_loop(const LoweredExprIR::LoweredLoopManager::LoweredLoopInfoPtr& loop_info,
                                     const LoweredExprPort& actual_port, const std::vector<LoweredExprPort>& target_ports, bool is_entry) {
    auto& ports = is_entry ? loop_info->m_entry_exprs : loop_info->m_exit_exprs;
    auto port_it = std::find(ports.begin(), ports.end(), actual_port);
    if (port_it == ports.end())
        return;
    port_it = ports.erase(port_it);
    ports.insert(port_it, target_ports.cbegin(), target_ports.cend());
}

bool LoadStoreInsertion::insert_load(LoweredExprIR& linear_ir,
                                     const LoweredExprIR::LoweredLoopManagerPtr& loop_manager,
                                     const LoweredExprPort& entry_point,
                                     LoweredExprIR::constExprIt loop_begin_pos, LoweredExprIR::constExprIt loop_end_pos) {
    const auto expr = entry_point.first;
    const auto port = entry_point.second;
    const auto node = expr->get_node();
    if (ov::is_type<op::Load>(node)) {
        return false;
    }

    const auto input_td = expr->get_inputs()[port];
    const auto parent_expr = linear_ir.get_expr_by_output(input_td);
    const auto parent = parent_expr->get_node();
    const auto parent_port = parent_expr->get_output_port_num(input_td);

    if (!ov::is_type<op::Buffer>(parent) && !ov::is_type<opset1::Parameter>(parent)) {
        return false;
    }

    const auto load_td = std::make_shared<TensorDescriptor>(input_td->get_tensor(),
                                                            input_td->get_subtensor(),
                                                            input_td->get_layout());
    const auto load = std::make_shared<op::Load>(parent->output(parent_port), m_vector_size);
    const auto load_outs = std::vector<TensorDescriptorPtr>{ load_td };
    const auto param_outs = std::vector<TensorDescriptorPtr>{ input_td };
    const auto load_expr = std::make_shared<LoweredExpr>(load, param_outs, load_outs);
    linear_ir.insert(std::find(loop_begin_pos, loop_end_pos, expr), load_expr);
    linear_ir.replace_input(expr, input_td, load_td);
    const auto new_entry_point = LoweredExprPort{load_expr, 0};
    // Copy Loop identifies
    const auto loop_identifies = expr->get_loop_identifies();
    load_expr->set_loop_identifies(loop_identifies);

    // Need to update all the corresponding Loops with the same Entry Point
    update_loops(loop_manager, loop_identifies, entry_point, {new_entry_point}, true);
    return true;
}

bool LoadStoreInsertion::insert_store(LoweredExprIR& linear_ir,
                                      const LoweredExprIR::LoweredLoopManagerPtr& loop_manager,
                                      const LoweredExprPort& exit_point,
                                      LoweredExprIR::constExprIt loop_begin_pos, LoweredExprIR::constExprIt loop_end_pos) {
    const auto expr = exit_point.first;
    const auto port = exit_point.second;
    const auto node = expr->get_node();
    if (ov::is_type<op::Store>(node)) {
        return false;
    }

    bool was_inserted = false;
    std::vector<LoweredExprPort> new_exit_exprs;
    const auto output_td = expr->get_outputs()[port];
    const auto child_exprs = linear_ir.get_exprs_by_input(output_td);
    const auto loop_identifies = expr->get_loop_identifies();
    auto store_pos = std::next(std::find(loop_begin_pos, linear_ir.cend(), expr));
    for (const auto& child_expr : child_exprs) {
        const auto child = child_expr->get_node();
        const auto port = child_expr->get_input_port_num(output_td);

        if (!ov::is_type<op::Buffer>(child) && !ov::is_type<opset1::Result>(child)) {
            continue;
        }

        const auto store_td = std::make_shared<TensorDescriptor>(output_td->get_tensor(),
                                                                 output_td->get_subtensor(),
                                                                 output_td->get_layout());
        auto store = std::make_shared<op::Store>(node->output(port), m_vector_size);
        const std::vector<TensorDescriptorPtr> parent_outs { output_td };
        const std::vector<TensorDescriptorPtr> store_outs { store_td };
        const auto store_expr = std::make_shared<LoweredExpr>(store, parent_outs, store_outs);
        linear_ir.insert(store_pos, store_expr);
        linear_ir.replace_input(child_expr, output_td, store_td);
        // Copy Loop identifies
        store_expr->set_loop_identifies(loop_identifies);
        // Update entry expressions.
        new_exit_exprs.push_back({store_expr, 0});
        was_inserted = true;
    }

    // Need to update all the corresponding Loops with the same Exit Point
    update_loops(loop_manager, loop_identifies, exit_point, new_exit_exprs, false);
    return was_inserted;
}

bool LoadStoreInsertion::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(itt::domains::SnippetsTransform, "Snippets::LoadStoreInsertion")

    bool modified = false;
    const auto& loop_manager = linear_ir.get_loop_manager();

    auto loop_begin_pos = linear_ir.cbegin();
    auto loop_end_pos = linear_ir.cend();
    std::vector<size_t> prev_expr_loops;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto expr = *expr_it;
        const auto &node = expr->get_node();
        if (ov::is_type<opset1::Parameter>(node) ||
            ov::is_type<opset1::Constant>(node) ||
            ov::is_type<opset1::Result>(node) ||
            ov::is_type<op::Brgemm>(node))
            continue;

        // Found Inner Loop
        const auto expr_loops = expr->get_loop_identifies();
        if (prev_expr_loops == expr_loops || expr_loops.empty()) {
            continue;
        }
        prev_expr_loops = expr_loops;
        const auto loop_depth = expr_loops.size();
        size_t loop_id = LoweredExpr::LOOP_NULL_ID;
        for (int i = loop_depth - 1; i >= 0; --i) {
            if (expr_loops[i] != LoweredExpr::LOOP_NULL_ID) {
                loop_id = expr_loops[i];
                break;
            }
        }

        const auto& loop_info = loop_manager->get(loop_id);
        const auto entry_exprs = loop_info->m_entry_exprs;
        const auto exit_exprs = loop_info->m_exit_exprs;
        LoweredLoopManager::get_loop_bounds(linear_ir, entry_exprs, exit_exprs, loop_begin_pos, loop_end_pos, loop_id);

        for (const auto& entry_point : entry_exprs) {
            modified |= insert_load(linear_ir, loop_manager, entry_point, loop_begin_pos, loop_end_pos);
        }
        for (const auto& exit_point : exit_exprs) {
            modified |= insert_store(linear_ir, loop_manager, exit_point, loop_begin_pos, loop_end_pos);
        }
    }

    return modified;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
