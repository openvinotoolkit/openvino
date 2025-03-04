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

// Specific value to mark ports which doesn't affect output shape of broadcastable ops.
// For example, ops with output scalar shape or Horizon ops.
static const size_t NOT_AFFECTING_PATH = SIZE_MAX;

static bool is_shape_broadcastable_op(const ExpressionPtr& expr) {
    return expr->get_node()->get_autob() != ov::op::AutoBroadcastType::NONE;
}

static bool is_not_affecting_op(const ExpressionPtr& expr) {
    const auto& node = expr->get_node();
    return ov::is_type_any_of<ov::snippets::op::HorizonMax,
                              ov::snippets::op::HorizonSum,
                              ov::snippets::op::ReduceMax,
                              ov::snippets::op::ReduceSum,
                              ov::snippets::op::VectorBuffer,
                              ov::snippets::op::BroadcastMove,
                              ov::snippets::op::Scalar>(node);
}

static bool is_affecting_op(const ExpressionPtr& expr) {
    const auto& node = expr->get_node();
    return ov::is_type_any_of<ov::snippets::op::Brgemm, ov::snippets::op::Reshape, ov::snippets::op::LoadReorder>(node);
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

    // Shape -> color
    std::map<VectorDims, size_t> colored_shapes;

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
            const auto& out_shape = expr->get_output_port_descriptor(out_idx)->get_shape();
            size_t current_color_path;
            if (colored_shapes.count(out_shape)) {
                current_color_path = colored_shapes.at(out_shape);
            } else if (!utils::is_dynamic_vdims(out_shape) && ov::shape_size(out_shape) == 1) {
                current_color_path = NOT_AFFECTING_PATH;
            } else {
                if (is_affecting_op(expr)) {
                    current_color_path = ++color_path;
                } else if (is_not_affecting_op(expr)) {
                    current_color_path = NOT_AFFECTING_PATH;
                } else if (is_shape_broadcastable_op(expr)) {
                    current_color_path = NOT_AFFECTING_PATH;
                    for (size_t in_idx = 0; in_idx < expr->get_input_count(); ++in_idx) {
                        const auto input_path = getInvariantPortShapePath(expr->get_input_port(in_idx));
                        current_color_path = merge_paths(current_color_path, input_path);
                    }
                } else {
                    current_color_path = expr->get_input_count() > 0 ? getInvariantPortShapePath(expr->get_input_port(0))
                                                                     : ++color_path;
                }

                if (!utils::is_dynamic_vdims(out_shape))
                    colored_shapes[out_shape] = current_color_path;
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
