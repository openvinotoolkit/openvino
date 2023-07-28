// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/load_movebroadcast_to_broadcastload.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool LoadMoveBroadcastToBroadcastLoad::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::LoadMoveBroadcastToBroadcastLoad")
    const auto& loop_manager = linear_ir.get_loop_manager();
    bool modified = false;

    for (auto expr_it = linear_ir.cbegin(); expr_it != linear_ir.cend(); expr_it++) {
        const auto& expr = *expr_it;
        const auto& op = expr->get_node();
        // Match on MoveBroadcast because MoveBroadcast is rare node in bodies
        if (const auto move_broadcast = ov::as_type_ptr<op::BroadcastMove>(op)) {
            const auto& interm_connector = expr->get_input_port_connector(0);
            const auto parent_expr = interm_connector->get_source().get_expr();
            const auto load = ov::as_type_ptr<op::Load>(parent_expr->get_node());
            if (!load)
                continue;

            // Cannot rewrite Broadcast + Load if load has more than 1 user
            // or more than one input, or if Broadcast has several inputs
            const auto load_consumers_inputs = interm_connector->get_consumers();
            size_t count = 0;
            for (const auto& consumer_expr_input : load_consumers_inputs) {
                const auto consumer = consumer_expr_input.get_expr()->get_node();
                if (!ov::is_type<op::LoopEnd>(consumer))
                    count++;
            }

            if (count > 1)
                continue;

            const auto& outshape = move_broadcast->get_output_partial_shape(0);
            const auto broadcastload = std::make_shared<snippets::op::BroadcastLoad>(load->input_value(0), outshape, load->get_offset());
            const auto move_consumers = expr->get_output_port_connector(0)->get_consumers();
            PortDescriptorUtils::set_port_descriptor_ptr(broadcastload->output(0), expr->get_output_port(0).get_descriptor_ptr()->clone());
            const auto broadcastload_expr = linear_ir.create_expression(broadcastload, { parent_expr->get_input_port_connector(0) });
            // Copy Loop identifies
            broadcastload_expr->set_loop_ids(parent_expr->get_loop_ids());
            // Update the corresponding Loops with
            loop_manager->update_loops_port(parent_expr->get_loop_ids(), parent_expr->get_input_port(0), {broadcastload_expr->get_input_port(0)}, true);

            const auto mv_expr_it = expr_it;
            const auto insertion_pos = std::next(expr_it);
            expr_it = linear_ir.insert(insertion_pos, broadcastload_expr);
            linear_ir.erase(linear_ir.find_before(mv_expr_it, parent_expr));
            linear_ir.erase(mv_expr_it);
            linear_ir.replace_input(move_consumers, broadcastload_expr->get_output_port_connector(0));
            modified |= true;
        }
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
