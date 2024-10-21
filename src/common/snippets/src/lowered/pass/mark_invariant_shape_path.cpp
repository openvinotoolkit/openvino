// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "snippets/lowered/pass/mark_invariant_shape_path.hpp"

#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

namespace {
static bool is_planar_layout(const std::vector<size_t>& order) {
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i] != i) return false;
    }
    return true;
}
static bool is_shape_invariant_op(const ExpressionPtr& expr) {
    if (expr->get_output_count() != 1)
        return false;
    auto is_invariant_ma_op = [](const ExpressionPtr& expr) {
        const auto& op = expr->get_node();
        return (ov::is_type<op::Load>(op) || ov::is_type<op::Store>(op)) &&
               is_planar_layout(expr->get_input_port_descriptor(0)->get_layout()) &&
               is_planar_layout(expr->get_output_port_descriptor(0)->get_layout());
    };
    const auto& node = expr->get_node();
    return is_invariant_ma_op(expr) ||
           ov::is_type<ov::snippets::lowered::BufferExpression>(expr) ||
           ov::op::util::is_unary_elementwise_arithmetic(node) ||
           ov::op::util::is_binary_elementwise_arithmetic(node) ||
           ov::op::util::is_binary_elementwise_comparison(node) ||
           ov::op::util::is_binary_elementwise_logical(node) ||
           ov::is_type<ov::op::v1::Select>(node) ||
           ov::is_type<ov::snippets::op::Fill>(node) ||
           ov::is_type<ov::snippets::op::ConvertTruncation>(node) ||
           ov::is_type<ov::snippets::op::ConvertSaturation>(node);
}
static bool is_sync_op(const ExpressionPtr& expr) {
    const auto& node = expr->get_node();
    const auto& ma = std::dynamic_pointer_cast<ov::snippets::modifier::MemoryAccess>(node);
    return ma && ma->is_full_memory_access_op(node) && expr->get_output_count() == 1;
}
}  // namespace

bool MarkInvariantShapePath::run(lowered::LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::MarkInvariantShapePath");

    size_t color_path = 0;

    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = *expr_it;
        if (ov::is_type<ov::snippets::op::LoopBase>(expr->get_node()))
            continue;

        for (size_t out_idx = 0; out_idx < expr->get_output_count(); ++out_idx) {
            size_t current_color_path;
            if (is_shape_invariant_op(expr)) {
                OPENVINO_ASSERT(initInvariantPortShapePaths(expr->get_input_port(0), current_color_path),
                                "Expected that input port is already marked");
            } else {
                current_color_path = color_path;
                color_path++;
            }
            const auto& output = expr->get_output_port_connector(out_idx);
            SetInvariantPortShapePath(output->get_source(), current_color_path);
            for (const auto& consumer : output->get_consumers()) {
                const auto& consumer_expr = consumer.get_expr();
                if (ov::is_type<ov::snippets::op::LoopEnd>(consumer_expr->get_node()))
                    continue;

                SetInvariantPortShapePath(consumer, current_color_path);
            }
        }
    }

    return color_path > 0;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
