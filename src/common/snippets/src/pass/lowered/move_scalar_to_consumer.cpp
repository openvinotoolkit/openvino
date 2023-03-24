// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/move_scalar_to_consumer.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

bool MoveScalarToConsumer::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::MoveScalarToConsumer")
    if (linear_ir.empty())
        return false;

    bool modified = false;
    // Visit expressions in reverse order, so we'll move Scalar to an already visited area.
    // This is needed to avoid extra hits, when we match to the same Scalar twice
    for (auto expr_it = linear_ir.rbegin(); expr_it != linear_ir.rend(); expr_it++) {
        const auto expr = expr_it->get();
        if (ov::is_type<op::Scalar>(expr->get_node())) {
            const auto& output = expr->get_outputs().front();
            const auto& consumers = linear_ir.get_exprs_by_input(output);
            OPENVINO_ASSERT(consumers.size() == 1, "Scalar expression is expected to have a single consumer");

            const auto& consumer_expr = consumers.begin()->expr;
            // Move something only if consumer is not already the next one (previous since the iterator is a reverse one)
            auto forward_it = std::prev(expr_it.base());
            if (consumer_expr != *std::next(forward_it)) {
                expr_it = std::prev(expr_it);  // save iterator before moving
                auto consumer_it = forward_it;
                while (*consumer_it != consumer_expr)
                    consumer_it++;
                linear_ir.move(forward_it, consumer_it);
                modified = true;
            }
        }
    }
    return modified;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
