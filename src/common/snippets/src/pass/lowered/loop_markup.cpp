// Copyright (C) 2023 Intel Corporationidentifies
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/loop_markup.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

LoopMarkup::LoopMarkup(size_t vector_size) : LinearIRTransformation(), m_vector_size(vector_size) {}

// Due to features of topological sort, some Constants (Scalars) may appear right after Parameters in
// sorted ops (so it's between Parameters and LoopBegin). Consequently, ScalarEmitters would be called
// outside the Loop, and only the first Loop iteration would yield correct data (assuming the vector reg
// assigned to scalar will get corrupted inside the loop body). To avoid such cases, we add Constants to the places in Linear IR
// in the corresponding Loops.
void LoopMarkup::scalars_markup(LoweredExprIR& linear_ir, std::vector<LoweredExprIR::constExprIt>& scalars_iterators) {
    for (const auto& scalar_it : scalars_iterators) {
        const auto scalar_expr = *scalar_it;
        const auto output_td = scalar_expr->get_outputs().front();
        const auto consumers = linear_ir.get_exprs_by_input(output_td);
        OPENVINO_ASSERT(consumers.size() == 1, "Scalar must have only one consumer!");
        const auto consumer = (*(consumers.begin())).first;
        scalar_expr->set_loop_identifies(consumer->get_loop_identifies());

        const auto consumer_it = std::find(scalar_it, linear_ir.cend(), consumer);
        OPENVINO_ASSERT(consumer_it != linear_ir.cend(), "Scalar consumer hasn't been found in linear IR!");
        if (scalar_it != std::prev(consumer_it)) {
            linear_ir.splice(consumer_it, scalar_it);
        }
    }
}

bool LoopMarkup::run(LoweredExprIR& linear_ir) {
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
               ov::is_type<opset1::Softmax>(node);  // Softmax is decomposed operation. The marking will be in decomposition pass
    };

    // Scalars cannot be a single node in Loop
    // So after common markup we should manually mark Scalars and
    // explicitly move to before the corresponding consumer
    std::vector<LoweredExprIR::constExprIt> scalars_iterators;

    for (auto expr_it = linear_ir.cbegin(); expr_it != linear_ir.cend(); expr_it++) {
        const auto expr = *expr_it;
        const auto& node = expr->get_node();
        if (ov::is_type<opset1::Constant>(node)) {
            scalars_iterators.push_back(expr_it);
            continue;
        }
        if (is_not_start_point(node))
            continue;
        if (ov::is_type<op::Brgemm>(node)) {
            loop_manager->skipped_marking(expr_it, std::next(expr_it), loop_depth);
            continue;
        }

        auto loop_begin_pos = expr_it;
        auto loop_end_pos = loop_begin_pos;

        const auto& outputs = expr->get_outputs();
        const auto& loop_inner_layout = outputs.front()->get_layout();
        const auto& loop_inner_subtensor = outputs.front()->get_subtensor();

        std::vector<LoweredExprPtr> body_exprs;
        bool is_inside = true;
        do {
            body_exprs.push_back(*loop_end_pos);
            loop_end_pos++;
            // If iterator is the last, we should finish Loop
            if (loop_end_pos == linear_ir.end())
                break;

            // If iterator is the last, we should finish Loop
            const auto& current_expr = *loop_end_pos;
            const auto& current_node = current_expr->get_node();
            if (ov::is_type<op::Brgemm>(current_node) ||
                ov::is_type<opset1::Softmax>(current_node) ||
                ov::is_type<opset1::Result>(current_node) ||
                ov::is_type<opset1::Constant>(current_node))
                break;

            // If the next expr isn't real customer of prev expr we should finish Loop
            const auto& ins = loop_end_pos->get()->get_inputs();
            auto connected = [&](const TensorDescriptorPtr& td) {return linear_ir.get_expr_by_output(td).first == body_exprs.back();};
            if (std::none_of(ins.begin(), ins.end(), connected))
                break;

            const auto& layout = ins.front()->get_layout();
            const auto& subtensor = ins.front()->get_subtensor();
            is_inside &= layout == loop_inner_layout && subtensor == loop_inner_subtensor;
        } while (is_inside);

        loop_manager->marking(linear_ir, loop_begin_pos, loop_end_pos, loop_depth, m_vector_size, body_exprs);
        expr_it = std::prev(loop_end_pos);
    }

    scalars_markup(linear_ir, scalars_iterators);

    return true;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
