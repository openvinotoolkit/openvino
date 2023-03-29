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

namespace {
auto get_inner_loop_id(const std::vector<size_t>& loop_ids) -> size_t {
    size_t inner_loop = LoweredExpr::LOOP_NULL_ID;
    for (int i = static_cast<int>(loop_ids.size()) - 1; i >= 0; --i) {
        if (loop_ids[i] != LoweredExpr::LOOP_NULL_ID) {
            inner_loop = loop_ids[i];
            break;
        }
    }
    return inner_loop;
}
} // namespace

using LoweredLoopManager = LoweredExprIR::LoweredLoopManager;
using LoweredLoopInfoPtr = LoweredLoopManager::LoweredLoopInfoPtr;

LoadStoreInsertion::LoadStoreInsertion(size_t vector_size) : m_vector_size(vector_size) {}

void LoadStoreInsertion::update_loops(const LoweredExprIR::LoweredLoopManagerPtr& loop_manager, const std::vector<size_t>& loop_ids,
                                      const LoweredExprPort& actual_port, const std::vector<LoweredExprPort>& target_ports, bool is_entry) {
    for (auto loop_id : loop_ids) {
        if (loop_id != LoweredExpr::LOOP_NULL_ID)
            update_loop(loop_manager->get_loop_info(loop_id), actual_port, target_ports, is_entry);
    }
}

void LoadStoreInsertion::update_loop(const LoweredExprIR::LoweredLoopManager::LoweredLoopInfoPtr& loop_info,
                                     const LoweredExprPort& actual_port, const std::vector<LoweredExprPort>& target_ports, bool is_entry) {
    auto& ports = is_entry ? loop_info->entry_exprs : loop_info->exit_exprs;
    auto port_it = std::find(ports.begin(), ports.end(), actual_port);
    if (port_it == ports.end())
        return;
    port_it = ports.erase(port_it);
    ports.insert(port_it, target_ports.cbegin(), target_ports.cend());
}

bool LoadStoreInsertion::insert_load(LoweredExprIR& linear_ir, const LoweredExprIR::constExprIt& data_expr_it) {
    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& data_expr = *data_expr_it;
    const auto& data_node = data_expr->get_node();
    const auto& output_td = data_expr->get_outputs().front();
    const auto consumer_inputs = linear_ir.get_exprs_by_input(output_td);

    bool was_inserted = false;
    for (const auto& consumer_input : consumer_inputs) {
        const auto& consumer_expr = consumer_input.expr;
        const auto port = consumer_input.port;
        const auto& consumer = consumer_expr->get_node();
        const auto ma = ov::as_type_ptr<op::MemoryAccess>(consumer);
        if (ma && ma->is_memory_access_input_port(port))
            return false;

        // Find Inner Loop
        const auto& loop_ids = consumer_expr->get_loop_ids();
        const auto inner_loop = get_inner_loop_id(loop_ids);
        OPENVINO_ASSERT(inner_loop != LoweredExpr::LOOP_NULL_ID, "Loop hasn't been found!");

        const auto load_td = std::make_shared<TensorDescriptor>(output_td->get_tensor(),
                                                                output_td->get_subtensor(),
                                                                output_td->get_layout());
        const auto load = std::make_shared<op::Load>(data_node->output(0), m_vector_size);
        const auto load_outs = std::vector<TensorDescriptorPtr>{ load_td };
        const auto param_outs = std::vector<TensorDescriptorPtr>{ output_td };
        const auto load_expr = std::make_shared<LoweredExpr>(load, param_outs, load_outs);
        linear_ir.insert(std::find(data_expr_it, linear_ir.cend(), consumer_expr), load_expr);
        linear_ir.replace_input(consumer_expr, port, load_td);
        // Copy Loop identifies
        load_expr->set_loop_ids(loop_ids);

        // Need to update all the corresponding Loops with the same Entry Point
        const auto prev_entry_point = consumer_input;
        const auto new_entry_point = LoweredExprPort::make_input(load_expr, 0);
        update_loops(loop_manager, loop_ids, prev_entry_point, {new_entry_point}, true);
        was_inserted = true;
    }

    return was_inserted;
}

bool LoadStoreInsertion::insert_store(LoweredExprIR& linear_ir, const LoweredExprIR::constExprIt& data_expr_it) {
    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& data_expr = *data_expr_it;
    const auto& input_td = data_expr->get_inputs().front();
    const auto parent_output = linear_ir.get_expr_by_output(input_td);
    const auto& parent_expr = parent_output.expr;
    const auto port = parent_output.port;
    const auto& parent = parent_expr->get_node();
    const auto ma = ov::as_type_ptr<op::MemoryAccess>(parent);
    if (ma && ma->is_memory_access_output_port(port))
        return false;

    // Find Inner Loop
    const auto& loop_ids = parent_expr->get_loop_ids();
    const auto inner_loop = get_inner_loop_id(loop_ids);
    OPENVINO_ASSERT(inner_loop != LoweredExpr::LOOP_NULL_ID, "Loop hasn't been found!");

    const auto store_td = std::make_shared<TensorDescriptor>(input_td->get_tensor(),
                                                             input_td->get_subtensor(),
                                                             input_td->get_layout());
    const auto store = std::make_shared<op::Store>(parent->output(port), m_vector_size);
    const auto store_outs = std::vector<TensorDescriptorPtr>{ store_td };
    const auto param_outs = std::vector<TensorDescriptorPtr>{ input_td };
    const auto store_expr = std::make_shared<LoweredExpr>(store, param_outs, store_outs);
    const auto& reverse_insertion_pos = std::find(std::reverse_iterator<LoweredExprIR::constExprIt>(data_expr_it), linear_ir.crend(), parent_expr);
    const auto& insertion_pos = reverse_insertion_pos.base();
    linear_ir.insert(insertion_pos, store_expr);
    linear_ir.replace_input(data_expr, 0, store_td);
    // Copy Loop identifies
    store_expr->set_loop_ids(loop_ids);

    // Need to update all the corresponding Loops with the same Exit Point
    const auto prev_exit_point = parent_output;
    // The previous exit point byt one output port can have several consumers that can be potential exit points
    // So we should verify on the possible future exit points
    const auto consumer_inputs = linear_ir.get_exprs_by_input(input_td);
    const auto should_be_saved = std::any_of(consumer_inputs.begin(), consumer_inputs.end(),
                                [](const LoweredExprPort& input_port) {
                                    const auto& node = input_port.expr->get_node();
                                    return ov::is_type<opset1::Result>(node) || ov::is_type<op::Buffer>(node);
                                });
    const auto new_exit_point = LoweredExprPort::make_output(store_expr, 0);
    const auto new_exit_points = should_be_saved ? std::vector<LoweredExprPort>{prev_exit_point, new_exit_point}
                                                 : std::vector<LoweredExprPort>{new_exit_point};
    update_loops(loop_manager, loop_ids, prev_exit_point, new_exit_points, false);
    return true;
}

bool LoadStoreInsertion::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::LoadStoreInsertion")

    bool modified = false;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto expr = *expr_it;
        const auto &node = expr->get_node();
        if (ov::is_type<opset1::Parameter>(node) || ov::is_type<op::Buffer>(node)) {
            modified |= insert_load(linear_ir, expr_it);
        }

        if (ov::is_type<opset1::Result>(node) || ov::is_type<op::Buffer>(node)) {
            modified |= insert_store(linear_ir, expr_it);
        }
    }

    return modified;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
