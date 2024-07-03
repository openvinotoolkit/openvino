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

bool LoadMoveBroadcastToBroadcastLoad::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::LoadMoveBroadcastToBroadcastLoad")
    bool modified = false;

    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& expr = *expr_it;
        const auto& op = expr->get_node();
        // Match on MoveBroadcast because MoveBroadcast is rare node in bodies
        if (const auto move_broadcast = ov::as_type_ptr<op::BroadcastMove>(op)) {
            const auto& interm_connector = expr->get_input_port_connector(0);
            const auto load_expr = interm_connector->get_source().get_expr();
            const auto load = ov::as_type_ptr<op::Load>(load_expr->get_node());
            if (!load ||
                 load->get_input_element_type(0) != load->get_output_element_type(0) ||
                 load_expr->get_input_port_descriptor(0)->get_shape() != load_expr->get_output_port_descriptor(0)->get_shape())
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

            const auto& load_parent_node = load_expr->get_input_port_connector(0)->get_source().get_expr()->get_node();
            const auto& outshape = move_broadcast->get_output_partial_shape(0);
            const auto broadcastload = std::make_shared<snippets::op::BroadcastLoad>(load_parent_node, *outshape.rbegin(), load->get_offset());
            // insert at position of load_expr. As BroadcastMove and Load will be removed, preserve expr_it.
            expr_it = std::next(expr_it);
            linear_ir.replace_with_node({ load_expr, expr }, broadcastload, load_expr->get_loop_ids(), linear_ir.find(load_expr));
            expr_it = std::prev(expr_it);
            modified |= true;
        }
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
