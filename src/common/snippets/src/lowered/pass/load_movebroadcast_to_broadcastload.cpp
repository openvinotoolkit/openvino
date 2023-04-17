// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/load_movebroadcast_to_broadcastload.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace lowered {
namespace pass {


bool LoadMoveBroadcastToBroadcastLoad::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::LoadMoveBroadcastToBroadcastLoad")
    bool modified = false;

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& op = (*expr_it)->get_node();
        // Match on MoveBroadcast because MoveBroadcast is rare node in bodies
        if (const auto move_broadcast = ov::as_type_ptr<op::BroadcastMove>(op)) {
            const auto interm_td = (*expr_it)->get_inputs().front();
            const auto parent_expr = linear_ir.get_expr_by_output(interm_td).expr;
            const auto load = ov::as_type_ptr<op::Load>(parent_expr->get_node());
            if (!load)
                continue;

            // Cannot rewrite Broadcast + Load if load has more than 1 user
            // or more than one input, or if Broadcast has several inputs
            const auto load_consumers_inputs = linear_ir.get_exprs_by_input(interm_td);
            size_t count = 0;
            for (const auto& consumer_expr_input : load_consumers_inputs) {
                const auto consumer = consumer_expr_input.expr->get_node();
                if (!ov::is_type<op::LoopEnd>(consumer))
                    count++;
            }

            if (count > 1)
                continue;

            auto outshape = move_broadcast->get_output_partial_shape(0);
            auto broadcastload = std::make_shared<snippets::op::BroadcastLoad>(load->input_value(0), outshape, load->get_offset());
            const auto in_td =  std::vector<TensorDescriptorPtr>{ parent_expr->get_inputs().front() };
            const auto out_td = std::vector<TensorDescriptorPtr>{ (*expr_it)->get_outputs().front() };
            const auto mv_expr_it = expr_it;
            const auto insertion_pos = std::next(expr_it);
            linear_ir.erase(std::find(linear_ir.begin(), mv_expr_it, parent_expr));
            linear_ir.erase(mv_expr_it);
            expr_it = linear_ir.insert(insertion_pos, std::make_shared<Expression>(broadcastload, in_td, out_td));
            modified |= true;
        }
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ngraph
