// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "snippets/op/buffer.hpp"
#include "transformations/snippets/tpp/op/modifiers.hpp"
#include "set_tpp_leading_dim.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace pass {
namespace {
using ExpressionPort = snippets::lowered::ExpressionPort;
bool is_planar_layout(const std::vector<size_t>& layout) {
    for (size_t i = 0; i < layout.size(); i++) {
        if (layout[i] != i)
            return false;
    }
    return true;
}

size_t get_leading_dim(ExpressionPort port) {
    auto has_connected_buffer = [](ExpressionPort port) {
        bool has_buffer = false;
        for (const auto& p : port.get_connected_ports()) {
            if (ov::is_type<snippets::op::Buffer>(p.get_expr()->get_node())) {
                OPENVINO_ASSERT(!has_buffer, "Only one Buffer can be connected to a TPP op");
                has_buffer = true;
            }
        }
        return has_buffer;
    };
    const auto& port_desc = port.get_descriptor_ptr();
    auto layout = port_desc->get_layout();
    auto shape = port_desc->get_shape();
    if (has_connected_buffer(port)) {
        shape = port_desc->get_subtensor();
        OPENVINO_ASSERT(is_planar_layout(layout), "Only planar layouts are supported for Buffers");
        const auto rank_diff = static_cast<int64_t>(layout.size()) - static_cast<int64_t>(shape.size());
        if (rank_diff > 0)
            layout.erase(layout.end() - rank_diff, layout.end());
    }

    OPENVINO_ASSERT(layout.empty() || (layout.back() == layout.size() - 1 && layout.size() == shape.size()),
            "BrgemmTppEmitter detected invalid layout values: check that this shape + layout combination is schedulable");
    const auto dim = [&]() -> size_t {
            switch (port.get_type()) {
            // Input shape is original, so we need to correctly read this data by order
            // Example:
            //      Original shape (shape) = [1, 49, 2, 23]
            //      Layout (transpose order) = [2, 0, 1, 3]
            //      Transposed shape = [2, 1, 49, 23]
            //      The leading dimension is equal to stride of shape[layout[3]] = 2 x 23
            case ExpressionPort::Type::Input :
                return layout[layout.size() - 2]; // `1` in example
            // Output shape is already transposed, we need to correctly write the data with original shape by the order
            // Example:
            //      Original transposed shape (shape) = [49, 2, 7, 39]
            //      Layout (transpose order) = [2, 0, 1, 3]
            //      Before leading dimension with index 3 there is dimension with index 2 in planar layout.
            //      Since we have non-planar layout, we have to find this before LD dim in transposed order.
            //      In layout 2nd idx is first element, it means, that the leading dimension is equal to stride of shape[0]
            case ExpressionPort::Type::Output :
                return std::distance(layout.cbegin(), std::find(layout.cbegin(), layout.cend(), layout.size() - 2)); // 0 in the example: shape[0] = 49
            default:
                OPENVINO_THROW("Unsupported Expression port type");
        }
    }();
    return std::accumulate(shape.cbegin() + dim + 1, shape.cend(), 1, std::multiplies<size_t>());
}

} // namespace

SetTPPLeadingDim::SetTPPLeadingDim() : Pass() {}

bool SetTPPLeadingDim::run(snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SetTPPLeadingDim")
    if (linear_ir.empty())
        return false;

    bool modified = false;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& expr = *expr_it;
        const auto& node = expr->get_node();
        auto tpp_expr = std::dynamic_pointer_cast<modifier::TensorProcessingPrimitive>(node);
        if (!tpp_expr)
            continue;

        OPENVINO_ASSERT(tpp_expr->is_full_memory_access_op(node), "TPP Op is expected to be MemoryAccess on all ports");

        for (size_t i = 0; i < expr->get_input_count(); i++) {
            const auto ld = get_leading_dim(expr->get_input_port(i));
            tpp_expr->set_input_stride(ld, i);
        }
        for (size_t i = 0; i < expr->get_output_count(); i++) {
            const auto ld = get_leading_dim(expr->get_output_port(i));
            tpp_expr->set_output_stride(ld, i);
        }
        modified = true;
    }

    return modified;
}


} // namespace pass
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
