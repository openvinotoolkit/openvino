// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/propagate_layout.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace lowered {
namespace pass {

bool PropagateLayout::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::PropagateLayout")
    const auto& io_ops = linear_ir.get_IO_ops();
    auto io_ops_it = io_ops.begin();
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        if (*expr_it == *io_ops_it) {
            const auto& expr = io_ops_it->get();
            io_ops_it++;
            const bool is_input = expr->get_type() == IOExpression::io_type::INPUT;
            const auto& tds = is_input ? expr->get_outputs() : expr->get_inputs();
            if (tds.size() != 1)
                OPENVINO_THROW("Parameter/Results should have exactly one output/input");
            const auto& target_td = tds[0];
            // If input - we should be looking downstream, if output - upstream
            if (is_input) {
                const auto& child_exprs_inputs = linear_ir.get_exprs_by_input(target_td);
                // Note that here we consider only the first child (which is usually load),
                // but often there is another child - LoopEnd
                std::vector<size_t> child_layout{};
                for (const auto& child_input : child_exprs_inputs) {
                    const auto child = child_input.expr;
                    const auto& n = child->get_node();
                    if (is_type<op::Load>(n) || is_type<op::BroadcastLoad>(n)) {
                        // Note: this limitation could be relaxed to multiple ops,
                        // but all of them must have the same shape and layout
                        if (!child_layout.empty() && child->get_outputs().front()->get_layout() != child_layout)
                            OPENVINO_THROW("All children of an input expression must have the same layout");
                        child_layout = child->get_outputs().front()->get_layout();
                    }
                }
                if (!child_layout.empty()) {
                    auto new_td = TensorDescriptor(target_td.get()->get_tensor(), target_td.get()->get_subtensor(),
                                                   child_layout);
                    (*target_td) = new_td;
                }
            }
        }
    }
return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ngraph
