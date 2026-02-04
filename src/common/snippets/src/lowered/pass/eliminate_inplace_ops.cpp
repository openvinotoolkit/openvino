// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/eliminate_inplace_ops.hpp"

#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/expression_port.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/op/fill.hpp"

namespace ov::snippets::lowered::pass {

EliminateInplaceOps::EliminateInplaceOps(IsInplaceFillCallback is_inplace_fill_callback)
    : m_is_inplace_fill_callback(std::move(is_inplace_fill_callback)) {
    OPENVINO_ASSERT(m_is_inplace_fill_callback, "Callback for inplace Fill detection must be provided");
}

bool EliminateInplaceOps::is_inplace_fill(const ExpressionPtr& fill_expr) const {
    const auto fill = ov::as_type_ptr<snippets::op::Fill>(fill_expr->get_node());
    if (!fill) {
        return false;
    }

    const auto offset = fill->get_offset();
    const auto element_size = fill->get_output_element_type(0).size();

    return m_is_inplace_fill_callback(offset, element_size);
}

void EliminateInplaceOps::eliminate_fill(LinearIR& linear_ir, const ExpressionPtr& fill_expr) {
    // Inplace Fill has one input and one output
    // We need to redirect all consumers of the Fill's output to use the Fill's input instead

    OPENVINO_ASSERT(fill_expr->get_input_count() == 1, "Fill should have exactly one input");
    OPENVINO_ASSERT(fill_expr->get_output_count() == 1, "Fill should have exactly one output");

    const auto& fill_input_connector = fill_expr->get_input_port_connector(0);
    const auto& fill_output_connector = fill_expr->get_output_port_connector(0);

    // Get all consumers of this Fill operation
    const auto consumers = fill_output_connector->get_consumers();

    // Redirect all consumers to use the input of Fill directly
    lowered::replace_input_port_connectors(consumers, fill_input_connector);

    // Remove Fill from the linear IR
    linear_ir.erase(linear_ir.find(fill_expr));
}

bool EliminateInplaceOps::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::EliminateInplaceOps");

    bool modified = false;

    // Iterate through all expressions and eliminate inplace Fill operations
    // We iterate from the end to avoid iterator invalidation issues when erasing
    auto expr_it = linear_ir.begin();
    while (expr_it != linear_ir.end()) {
        const auto& expr = *expr_it;

        if (is_inplace_fill(expr)) {
            eliminate_fill(linear_ir, expr);
            modified = true;
            // After erasing, we need to reset the iterator
            expr_it = linear_ir.begin();
        } else {
            ++expr_it;
        }
    }

    return modified;
}

}  // namespace ov::snippets::lowered::pass
