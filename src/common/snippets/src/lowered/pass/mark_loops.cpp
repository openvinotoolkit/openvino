// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/mark_loops.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

MarkLoops::MarkLoops(size_t vector_size) : RangedPass(), m_vector_size(vector_size) {}

bool MarkLoops::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::MarkLoops")
    const auto& lowering_config = linear_ir.get_config();
    const auto& loop_manager = linear_ir.get_loop_manager();
    auto loop_depth = lowering_config.m_loop_depth;

    // Parameters, Results or Constants are ignored. They can't be in Loops
    auto is_loop_outside_op = [](const std::shared_ptr<ov::Node>& node) {
        return ov::is_type<ov::op::v0::Result>(node) ||
               ov::is_type<ov::op::v0::Constant>(node) ||
               ov::is_type<ov::op::v0::Parameter>(node) ||
               ov::is_type<op::RankNormalization>(node) ||
               ov::is_type<op::Reshape>(node);
    };

    auto are_conflicted = [](const ExpressionPort& lhs, const ExpressionPort& rhs) {
        const auto& lhs_desc = lhs.get_descriptor_ptr();
        const auto& rhs_desc = rhs.get_descriptor_ptr();
        return lhs_desc->get_subtensor() != rhs_desc->get_subtensor() ||
               lhs_desc->get_layout() != rhs_desc->get_layout();
    };

    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto expr = *expr_it;
        const auto& node = expr->get_node();
        if (is_loop_outside_op(node))
            continue;

        auto loop_begin_pos = expr_it;
        auto loop_end_pos = loop_begin_pos;

        bool collapse = true;
        do {
            const auto& prev_expr = *loop_end_pos;
            loop_end_pos++;
            // If iterator is the last, we should finish Loop
            if (loop_end_pos == end)
                break;

            // If iterator is the last, we should finish Loop
            const auto& current_expr = *loop_end_pos;
            if (is_loop_outside_op(current_expr->get_node()))
                break;

            // We finish Loop if
            //  - the next expr isn't real consumer
            //  - the is conflict between the corresponding ports
            bool is_connected = false;
            bool is_conflicted = false;
            for (size_t i = 0; i < current_expr->get_input_count(); ++i) {
                const auto current_port = current_expr->get_input_port(i);
                const auto source_port = current_expr->get_input_port_connector(i)->get_source();
                if (source_port.get_expr() == prev_expr) {
                    if (are_conflicted(current_port, source_port)) {
                        is_conflicted = true;
                        break;
                    }
                    is_connected = true;
                }
            }
            collapse = is_connected && !is_conflicted;
        } while (collapse);

        loop_manager->mark_loop(loop_begin_pos, loop_end_pos, loop_depth, m_vector_size);
        expr_it = std::prev(loop_end_pos);
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
