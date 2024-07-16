// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/move_result_out_of_loop.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool MoveResultOutOfLoop::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::MoveResultOutOfLoop")
    if (linear_ir.empty())
        return false;

    bool modified = false;
    const auto loop_manager = linear_ir.get_loop_manager();
    // Visit expressions in reverse order, so we'll move Result to an already visited area.
    // This is needed to avoid extra hits, when we match to the same Result twice
    for (auto expr_it = linear_ir.crbegin(); expr_it != linear_ir.crend(); expr_it++) {
        const auto& result_it = std::prev(expr_it.base());
        const auto& expr = *expr_it;
        const auto& node = expr->get_node();
        if (!ov::is_type<ov::op::v0::Result>(node)) {
            continue;
        }

        const auto& shape_infer_seq = utils::get_first_parent_shape_infer_expr_seq(expr);
        const auto& data_expr = shape_infer_seq.empty() ? expr : shape_infer_seq.back();
        const auto& parent_expr = data_expr->get_input_port_connector(0)->get_source().get_expr();
        const auto& parent_loop_ids = parent_expr->get_loop_ids();

        // Parent is out of Loop: just verify that Result is after Parent
        if (parent_loop_ids.empty()) {
            const auto parent_it = std::find(result_it, linear_ir.cend(), parent_expr);
            // If Parent is found after Result, we should move Result
            if (parent_it != linear_ir.cend()) {
                const auto insertion_pos = std::next(parent_it);
                expr_it = std::prev(expr_it);  // save iterator before moving
                linear_ir.move(result_it, insertion_pos);
                modified = true;
            }
        } else {
            const auto loop_bounds = loop_manager->get_loop_bounds(linear_ir, *(parent_loop_ids.cbegin()));
            const auto loop_end_pos = loop_bounds.second;
            // If the Result isn't found after Outer LoopEnd, need to move it to there
            if (std::find(loop_end_pos, linear_ir.cend(), expr) == linear_ir.cend()) {
                expr_it = std::prev(expr_it);  // save iterator before moving
                linear_ir.move(result_it, loop_end_pos);
                modified = true;
            }
        }

        // The Result is executed out of Loop
        expr->set_loop_ids({});

        // Move ShapeInfer ops to the Result
        for (const auto& shape_infer_expr : shape_infer_seq) {
            const auto it = std::find(linear_ir.cbegin(), linear_ir.cend(), shape_infer_expr);
            linear_ir.move(it, result_it);
            shape_infer_expr->set_loop_ids({});
        }
    }

    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
