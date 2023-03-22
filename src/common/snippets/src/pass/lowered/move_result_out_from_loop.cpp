// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/move_result_out_of_loop.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

bool MoveResultOutOfLoop::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::MoveResultOutOfLoop")
    if (linear_ir.empty())
        return false;

    bool modified = false;
    // Visit expressions in reverse order, so we'll move Result to an already visited area.
    // This is needed to avoid extra hits, when we match to the same Result twice
    std::deque<LoweredExprIR::constExprIt> loop_parenthesis;
    for (auto expr_it = linear_ir.crbegin(); expr_it != linear_ir.crend(); expr_it++) {
        const auto expr = expr_it->get();
        const auto node = expr->get_node();
        if (ov::is_type<op::LoopEnd>(node)) {
            loop_parenthesis.push_back(std::prev(expr_it.base()));
            continue;
        } else if (ov::is_type<op::LoopBegin>(node)) {
            loop_parenthesis.pop_back();
            continue;
        } else if (ov::is_type<opset1::Result>(node)) {
            // If the deque of parenthesis is empty, it's mean that Result is already out from Loop
            if (loop_parenthesis.empty())
                continue;

            // Get the outer LoopEnd iterator
            const auto insertion_pos = std::next(loop_parenthesis.front());
            const auto result_it = std::prev(expr_it.base());
            expr_it = std::prev(expr_it);  // save iterator before moving
            linear_ir.move(result_it, insertion_pos);
            modified = true;
        }
    }

    return modified;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
