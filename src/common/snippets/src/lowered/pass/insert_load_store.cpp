// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_load_store.hpp"

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

InsertLoadStore::InsertLoadStore(size_t vector_size) : m_vector_size(vector_size) {}

size_t InsertLoadStore::get_count(const PortDescriptorPtr& port_desc) const {
    const auto layout = port_desc->get_layout();
    const auto shape = port_desc->get_shape();
    // Find last dimension by layout
    const auto last_dim_idx = std::find(layout.begin(), layout.end(), layout.size() - 1);
    OPENVINO_ASSERT(last_dim_idx != layout.end() && *last_dim_idx < shape.size(), "Load/Store expression have incorrect layout");
    const auto dim = shape[*last_dim_idx];
    return dim == 1 ? 1 : m_vector_size;
}

bool InsertLoadStore::insert_load(LinearIR& linear_ir, const LinearIR::constExprIt& data_expr_it) {
    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& data_expr = *data_expr_it;
    const auto& data_node = data_expr->get_node();
    const auto& output_connector = data_expr->get_output_port_connector(0);
    const auto consumer_inputs = output_connector->get_consumers();

    bool was_inserted = false;
    for (const auto& consumer_input : consumer_inputs) {
        const auto& consumer_expr = consumer_input.get_expr();
        const auto port = consumer_input.get_index();
        const auto& consumer = consumer_expr->get_node();
        const auto ma = ov::as_type_ptr<op::MemoryAccess>(consumer);
        if (ma && ma->is_memory_access_input_port(port))
            return false;

        const auto loop_ids = consumer_expr->get_loop_ids();
        const auto load = std::make_shared<op::Load>(data_node->output(0), get_count(data_expr->get_output_port_descriptor(0)));
        PortDescriptorUtils::set_port_descriptor_ptr(load->output(0), consumer_input.get_descriptor_ptr()->clone());
        const auto load_expr = linear_ir.create_expression(load, {output_connector});
        linear_ir.insert(linear_ir.find_after(data_expr_it, consumer_expr), load_expr);
        linear_ir.replace_input(consumer_input, load_expr->get_output_port_connector(0));
        // Copy Loop identifies
        load_expr->set_loop_ids(loop_ids);

        // Need to update all the corresponding Loops with the same Entry Point
        const auto prev_entry_point = consumer_input;
        const auto new_entry_point = load_expr->get_input_port(0);
        loop_manager->update_loops_port(loop_ids, prev_entry_point, {new_entry_point}, true);
        was_inserted = true;
    }

    return was_inserted;
}

bool InsertLoadStore::insert_store(LinearIR& linear_ir, const LinearIR::constExprIt& data_expr_it) {
    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& data_expr = *data_expr_it;
    const auto& input_connector = data_expr->get_input_port_connector(0);
    const auto& parent_output = input_connector->get_source();
    const auto& parent_expr = parent_output.get_expr();
    const auto port = parent_output.get_index();
    const auto& parent = parent_expr->get_node();
    const auto ma = ov::as_type_ptr<op::MemoryAccess>(parent);
    if (ma && ma->is_memory_access_output_port(port))
        return false;

    const auto loop_ids = parent_expr->get_loop_ids();
    const auto store = std::make_shared<op::Store>(parent->output(port), get_count(data_expr->get_input_port_descriptor(0)));
    PortDescriptorUtils::set_port_descriptor_ptr(store->output(0), parent_output.get_descriptor_ptr()->clone());
    const auto store_expr = linear_ir.create_expression(store, {input_connector});
    const auto& insertion_pos = linear_ir.find_after(std::reverse_iterator<LinearIR::constExprIt>(data_expr_it), parent_expr).base();
    linear_ir.insert(insertion_pos, store_expr);
    linear_ir.replace_input(data_expr->get_input_port(0), store_expr->get_output_port_connector(0));
    // Copy Loop identifies
    store_expr->set_loop_ids(loop_ids);

    // Need to update all the corresponding Loops with the same Exit Point
    const auto prev_exit_point = parent_output;
    // The previous exit point but one output port can have several consumers that can be potential exit points
    // So we should verify on the possible future exit points
    const auto consumer_inputs = input_connector->get_consumers();
    const auto should_be_saved = std::any_of(consumer_inputs.begin(), consumer_inputs.end(),
                                [&data_expr](const ExpressionPort& input_port) {
                                    const auto expr = input_port.get_expr();
                                    // Skip the current data expr since the input of the expr is changed to Store expr
                                    if (expr == data_expr)
                                        return false;
                                    const auto& node = expr->get_node();
                                    return ov::is_type<ov::op::v0::Result>(node) || ov::is_type<op::Buffer>(node);
                                });
    const auto new_exit_point = store_expr->get_output_port(0);
    const auto new_exit_points = should_be_saved ? std::vector<ExpressionPort>{prev_exit_point, new_exit_point}
                                                 : std::vector<ExpressionPort>{new_exit_point};
    loop_manager->update_loops_port(loop_ids, prev_exit_point, new_exit_points, false);
    return true;
}

bool InsertLoadStore::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertLoadStore")

    bool modified = false;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto expr = *expr_it;
        const auto& node = expr->get_node();
        if (ov::is_type<ov::op::v0::Parameter>(node)) {
            modified |= insert_load(linear_ir, expr_it);
            continue;
        }
        if (ov::is_type<ov::op::v0::Result>(node)) {
            modified |= insert_store(linear_ir, expr_it);
            continue;
        }
        if (auto buffer = ov::as_type_ptr<op::Buffer>(node)) {
            modified |= insert_load(linear_ir, expr_it);
            if (buffer->is_intermediate_memory())
                modified |= insert_store(linear_ir, expr_it);
            continue;
        }
    }

    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
