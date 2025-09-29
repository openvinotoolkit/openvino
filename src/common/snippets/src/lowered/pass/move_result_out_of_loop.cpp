// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/move_result_out_of_loop.hpp"

#include <algorithm>
#include <iterator>

#include "openvino/core/type.hpp"
#include "openvino/op/result.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"

namespace ov::snippets::lowered::pass {

bool MoveResultOutOfLoop::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::MoveResultOutOfLoop")
    if (linear_ir.empty()) {
        return false;
    }

    bool modified = false;
    const auto loop_manager = linear_ir.get_loop_manager();
    // Visit expressions in reverse order, so we'll move Result to an already visited area.
    // This is needed to avoid extra hits, when we match to the same Result twice
    for (auto expr_it = linear_ir.crbegin(); expr_it != linear_ir.crend(); expr_it++) {
        const auto& forward_it = std::prev(expr_it.base());
        const auto& expr = *expr_it;
        const auto& node = expr->get_node();
        if (!ov::is_type<ov::op::v0::Result>(node)) {
            continue;
        }

        const auto& parent_expr = expr->get_input_expr_ptr(0);
        const auto& parent_loop_ids = parent_expr->get_loop_ids();

        // Parent is out of Loop: just verify that Result is after Parent
        if (parent_loop_ids.empty()) {
            const auto parent_it = std::find(forward_it, linear_ir.cend(), parent_expr);
            // If Parent is found after Result, we should move Result
            if (parent_it != linear_ir.cend()) {
                const auto insertion_pos = std::next(parent_it);
                const auto result_it = forward_it;
                expr_it = std::prev(expr_it);  // save iterator before moving
                linear_ir.move(result_it, insertion_pos);
                modified = true;
            }
            // The Result is executed out of Loop
            expr->set_loop_ids({});
            continue;
        }

        const auto loop_bounds = loop_manager->get_loop_bounds(linear_ir, *(parent_loop_ids.cbegin()));
        const auto loop_end_pos = loop_bounds.second;
        // If the Result isn't found after Outer LoopEnd, need to move it to there
        if (std::find(loop_end_pos, linear_ir.cend(), expr) == linear_ir.cend()) {
            expr_it = std::prev(expr_it);  // save iterator before moving
            linear_ir.move(forward_it, loop_end_pos);
            modified = true;
        }
        // The Result is executed out of Loop
        expr->set_loop_ids({});
    }

    return modified;
}

}  // namespace ov::snippets::lowered::pass
