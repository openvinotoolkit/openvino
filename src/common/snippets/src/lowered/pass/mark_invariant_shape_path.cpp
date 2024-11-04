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

// Specific value to mark ports which does't affect output shape of broadcastable ops.
// For example, ops with output scalar shape or Horizon ops.
static const size_t NOT_AFFECTING_PATH = SIZE_MAX;

static bool is_passed_through_op(const ExpressionPtr& expr) {
    if (expr->get_input_count() != 1 || expr->get_output_count() != 1)
        return false;

    auto is_invariant_ma_op = [](const ExpressionPtr& expr) {
        const auto& op = expr->get_node();
        return (ov::is_type<op::Load>(op) || ov::is_type<op::Store>(op)) &&
               utils::is_planar_layout(expr->get_input_port_descriptor(0)->get_layout()) &&
               utils::is_planar_layout(expr->get_output_port_descriptor(0)->get_layout());
    };
    const auto& node = expr->get_node();
    return is_invariant_ma_op(expr) ||
           ov::is_type<ov::snippets::lowered::BufferExpression>(expr) ||
           ov::op::util::is_unary_elementwise_arithmetic(node) ||
           ov::is_type<ov::snippets::op::Fill>(node) ||
           ov::is_type<ov::snippets::op::ConvertTruncation>(node) ||
           ov::is_type<ov::snippets::op::ConvertSaturation>(node);
}

static bool is_shape_broadcastable_op(const ExpressionPtr& expr) {
    const auto& node = expr->get_node();
    return ov::op::util::is_binary_elementwise_arithmetic(node) ||
           ov::op::util::is_binary_elementwise_comparison(node) ||
           ov::op::util::is_binary_elementwise_logical(node) ||
           ov::is_type<ov::op::v1::Select>(node);
}

static bool is_not_affecting_op(const ExpressionPtr& expr) {
    const auto& node = expr->get_node();
    return ov::is_type<ov::snippets::op::HorizonMax>(node) ||
           ov::is_type<ov::snippets::op::HorizonSum>(node) ||
           ov::is_type<ov::snippets::op::VectorBuffer>(node) ||
           ov::is_type<ov::snippets::op::BroadcastMove>(node);
}
}  // namespace

size_t MarkInvariantShapePath::getInvariantPortShapePath(const ExpressionPort& port) {
    auto& rt = get_rt_info(port);
    const auto rinfo = rt.find("InvariantShapePath");
    OPENVINO_ASSERT(rinfo != rt.end(), "Invariant path for this expression port has not been marked!");
    return rinfo->second.as<size_t>();
}

void MarkInvariantShapePath::SetInvariantPortShapePath(const ExpressionPort& port, size_t value) {
    OPENVINO_ASSERT(port.get_type() == ExpressionPort::Output, "SetInvariantPortShapePath can be used only for output port");
    auto& rt = get_rt_info(port);
    rt["InvariantShapePath"] = value;
}

ov::RTMap& MarkInvariantShapePath::get_rt_info(const ExpressionPort& port) {
    const auto& source_port = port.get_type() == ExpressionPort::Input ? port.get_port_connector_ptr()->get_source() : port;
    const auto& node = source_port.get_expr()->get_node();
    const auto port_idx = source_port.get_index();
    OPENVINO_ASSERT(port_idx < node->get_output_size(), "Node has incompatible port count with the expression");
    return node->output(port_idx).get_rt_info();
}

bool MarkInvariantShapePath::run(lowered::LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::MarkInvariantShapePath");

    bool modified = false;
    size_t color_path = 0;

    auto merge_paths = [&color_path](size_t lhs, size_t rhs) {
        if (lhs == rhs || rhs == NOT_AFFECTING_PATH) return lhs;
        if (lhs == NOT_AFFECTING_PATH) return rhs;
        return ++color_path;
    };

    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = *expr_it;
        if (ov::is_type<ov::snippets::op::LoopBase>(expr->get_node()))
            continue;

        for (size_t out_idx = 0; out_idx < expr->get_output_count(); ++out_idx) {
            size_t current_color_path;
            if (is_shape_broadcastable_op(expr)) {
                current_color_path = NOT_AFFECTING_PATH;
                for (size_t in_idx = 0; in_idx < expr->get_input_count(); ++in_idx) {
                    const auto input_path = getInvariantPortShapePath(expr->get_input_port(in_idx));
                    current_color_path = merge_paths(current_color_path, input_path);
                }
            } else if (is_passed_through_op(expr)) {
                current_color_path = getInvariantPortShapePath(expr->get_input_port(0));
            } else if (is_not_affecting_op(expr)) {
                current_color_path = NOT_AFFECTING_PATH;
            } else {
                current_color_path = ++color_path;
            }

            SetInvariantPortShapePath(expr->get_output_port(out_idx), current_color_path);
            modified = true;
        }
    }

    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
