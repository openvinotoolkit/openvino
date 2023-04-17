// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/loop_markup.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace lowered {
namespace pass {

LoopMarkup::LoopMarkup(size_t vector_size) : Transformation(), m_vector_size(vector_size) {}

bool LoopMarkup::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::LoopMarkup")
    if (linear_ir.empty())
        return false;

    const auto& lowering_config = linear_ir.get_config();
    const auto& loop_manager = linear_ir.get_loop_manager();
    auto loop_depth = lowering_config.m_loop_depth;

    // Parameters Results or Constants are ignored. They can't be used as a loop starting point
    auto is_not_start_point = [](const std::shared_ptr<ov::Node>& node) {
        return ov::is_type<opset1::Result>(node) ||
               ov::is_type<opset1::Constant>(node) ||
               ov::is_type<opset1::Parameter>(node) ||
               ov::is_type<opset1::Softmax>(node);  // Softmax is decomposed operation. The marking is in decomposition pass
    };

    for (auto expr_it = linear_ir.cbegin(); expr_it != linear_ir.cend(); expr_it++) {
        const auto expr = *expr_it;
        const auto& node = expr->get_node();
        if (is_not_start_point(node))
            continue;

        auto loop_begin_pos = expr_it;
        auto loop_end_pos = loop_begin_pos;

        const auto& outputs = expr->get_outputs();
        const auto& loop_inner_layout = outputs.front()->get_layout();
        const auto& loop_inner_subtensor = outputs.front()->get_subtensor();
        const bool loop_is_outside = expr->is_outside_loop();
        const bool loop_is_inside = !loop_is_outside;

        bool current_is_outside = loop_is_outside;
        bool current_is_inside = loop_is_inside;
        do {
            const auto& prev_expr = *loop_end_pos;
            loop_end_pos++;
            // If iterator is the last, we should finish Loop
            if (loop_end_pos == linear_ir.end())
                break;

            // If iterator is the last, we should finish Loop
            const auto& current_expr = *loop_end_pos;
            const auto& current_node = current_expr->get_node();
            if (ov::is_type<opset1::Softmax>(current_node) ||  // Softmax is marked in decomposition
                ov::is_type<opset1::Result>(current_node) ||
                ov::is_type<opset1::Constant>(current_node))
                break;

            const auto& ins = loop_end_pos->get()->get_inputs();
            current_is_inside = std::all_of(ins.begin(), ins.end(),
                                          [&loop_inner_layout, &loop_inner_subtensor](const TensorDescriptorPtr& td) {
                                              return td->get_layout() == loop_inner_layout &&
                                                     td->get_subtensor() == loop_inner_subtensor; });
            // If the next expr isn't real customer of prev expr we should finish Loop
            auto connected = [&](const TensorDescriptorPtr& td) {return linear_ir.get_expr_by_output(td).expr == prev_expr;};
            if (current_is_inside && std::none_of(ins.begin(), ins.end(), connected))
                break;

            current_is_outside = current_expr->is_outside_loop();
        } while (current_is_inside == loop_is_inside && current_is_outside == loop_is_outside);

        if (loop_is_inside)
            loop_manager->mark_loop(linear_ir, loop_begin_pos, loop_end_pos, loop_depth, m_vector_size);
        else if (loop_is_outside)
            loop_manager->skipped_mark(loop_begin_pos, loop_end_pos, loop_depth);

        expr_it = std::prev(loop_end_pos);
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ngraph
