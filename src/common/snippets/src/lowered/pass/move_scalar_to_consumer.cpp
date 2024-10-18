// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/move_scalar_to_consumer.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool MoveScalarToConsumer::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::MoveScalarToConsumer")
    if (linear_ir.empty())
        return false;

    bool modified = false;
    // Visit expressions in reverse order, so we'll move Scalar to an already visited area.
    // This is needed to avoid extra hits, when we match to the same Scalar twice
    for (auto expr_it = linear_ir.rbegin(); expr_it != linear_ir.rend(); expr_it++) {
        const auto expr = expr_it->get();
        if (ov::is_type<op::Scalar>(expr->get_node())) {
            const auto consumers = expr->get_output_port_connector(0)->get_consumers();
            OPENVINO_ASSERT(consumers.size() == 1, "Scalar expression is expected to have a single consumer");

            const auto& consumer_expr = consumers.begin()->get_expr();
            // Move something only if
            //  - Consumer is not already the next one (previous since the iterator is a reverse one)
            //  - The next operation is not already a Scalar (since it was just moved there on the previous iteration)
            auto forward_it = std::prev(expr_it.base());
            const auto& next_expr = *std::next(forward_it);
            if (consumer_expr != next_expr && !ov::is_type<op::Scalar>(next_expr->get_node())) {
                expr_it = std::prev(expr_it);  // save iterator before moving
                auto consumer_it = forward_it;
                while (*consumer_it != consumer_expr)
                    consumer_it++;
                linear_ir.move(forward_it, consumer_it);
                modified = true;
            }
            expr->set_loop_ids(consumer_expr->get_loop_ids());
        }
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
