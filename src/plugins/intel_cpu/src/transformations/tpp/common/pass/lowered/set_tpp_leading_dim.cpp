// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "set_tpp_leading_dim.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/tpp/common/op/modifiers.hpp"

namespace ov::intel_cpu::tpp::pass {
namespace {
using ExpressionPort = snippets::lowered::ExpressionPort;
using LoopPort = snippets::lowered::LoopPort;
// Note: Buffer is directly connected to the port if it remains in the same loops with the port's expression
//  Directly connected Buffers store data densely, so strides are defined by subternsor dims
//  Indirectly connected Buffers (with loops between the expr and Buffer) store data according
//  to their shape and layout
bool has_directly_connected_buffer(const ExpressionPort& port, const snippets::lowered::LoopManagerPtr& loop_mngr) {
    auto accepted_loops = [&loop_mngr, &port](const std::vector<size_t>& orig, const std::vector<size_t>& connect) {
        size_t connect_idx = 0;
        auto pred = [&port](const LoopPort& loop_port) {
            return *loop_port.get_expr_port() == port;
        };
        for (const auto orig_loop : orig) {
            if (connect_idx < connect.size() && orig_loop == connect[connect_idx]) {
                connect_idx++;
                continue;
            }
            // Note that orig expression can have some extra loops (compared to connect)
            // as long as the port is the loop entry/exit, and it is not incremented.
            // This is the case for Brgemm K-blocking loops, for example.
            const auto loop_info = loop_mngr->get_loop_info(orig_loop);
            const auto& border_points = port.get_type() == ExpressionPort::Type::Input ? loop_info->get_input_ports()
                                                                                       : loop_info->get_output_ports();
            const auto& found = std::find_if(border_points.begin(), border_points.end(), pred);
            if (found == border_points.end() || found->is_incremented())
                return false;
        }
        return true;
    };
    bool has_buffer = false;
    const auto& orig_loop_ids = port.get_expr()->get_loop_ids();
    for (const auto& p : port.get_connected_ports()) {
        const auto& connected_expr = p.get_expr();
        if (ov::is_type<snippets::op::Buffer>(connected_expr->get_node()) &&
            accepted_loops(orig_loop_ids, connected_expr->get_loop_ids())) {
            OPENVINO_ASSERT(!has_buffer, "Only one Buffer can be connected to a TPP op");
            has_buffer = true;
        }
    }
    return has_buffer;
}

size_t get_leading_dim(ExpressionPort port, const snippets::lowered::LoopManagerPtr& loop_mngr) {
    const auto& port_desc = port.get_descriptor_ptr();
    auto layout = port_desc->get_layout();
    auto shape = port_desc->get_shape();
    auto subtensor = port_desc->get_subtensor();
    // Some expressions (e.g. ReduceMax/ReduceSum) allow for FULL_DIM values in subtensor.
    // Here we should replace them with actual dim values before calculating strides & offsets.
    bool full_dim_substituted = false;
    for (size_t i = 1; i <= subtensor.size(); i++) {
        const auto idx = subtensor.size() - i;
        if (ov::snippets::utils::is_full_dim_value(subtensor[idx])) {
            // the reason that we don't support FULL_DIM substitution for an arbitrary layout is that
            // the layout and subtersor can (and usually do) have different ranks
            full_dim_substituted = true;
            subtensor[idx] = shape[shape.size() - i];
        }
    }
    OPENVINO_ASSERT(!full_dim_substituted || ov::snippets::utils::is_planar_layout(layout),
                    "Only planar layouts are supported for FULL_DIM substitution");

    if (has_directly_connected_buffer(port, loop_mngr)) {
        shape = port_desc->get_subtensor();
        OPENVINO_ASSERT(ov::snippets::utils::is_planar_layout(layout), "Only planar layouts are supported for Buffers");
        const auto rank_diff = static_cast<int64_t>(layout.size()) - static_cast<int64_t>(shape.size());
        if (rank_diff > 0) {
            layout.erase(layout.end() - rank_diff, layout.end());
        }
    }

    OPENVINO_ASSERT(layout.empty() || (layout.back() == layout.size() - 1 && layout.size() == shape.size()),
                    "get_leading_dim detected invalid layout values: check shape + layout combination");
    const auto dim = [&]() -> size_t {
        switch (port.get_type()) {
        // Input shape is original, so we need to correctly read this data by order
        // Example:
        //      Original shape (shape) = [1, 49, 2, 23]
        //      Layout (transpose order) = [2, 0, 1, 3]
        //      Transposed shape = [2, 1, 49, 23]
        //      The leading dimension is equal to stride of shape[layout[3]] = 2 x 23
        case ExpressionPort::Type::Input:
            return snippets::utils::get_input_dim_idx(layout, 1);  // `1` in example
        // Output shape is already transposed, we need to correctly write the data with original shape by the order
        // Example:
        //      Original transposed shape (shape) = [49, 2, 7, 39]
        //      Layout (transpose order) = [2, 0, 1, 3]
        //      Before leading dimension with index 3 there is dimension with index 2 in planar layout.
        //      Since we have non-planar layout, we have to find this before LD dim in transposed order.
        //      In layout 2nd idx is first element, it means, that the leading dimension is equal to stride of shape[0]
        case ExpressionPort::Type::Output:
            return snippets::utils::get_output_dim_idx(layout, 1);  // 0 in the example: shape[0] = 49
        default:
            OPENVINO_THROW("Unsupported Expression port type");
        }
    };
    return layout.size() == 1 ? shape.back()
                              : std::accumulate(shape.cbegin() + dim() + 1, shape.cend(), 1, std::multiplies<>());
}

}  // namespace

SetTPPLeadingDim::SetTPPLeadingDim() : RangedPass() {}

bool SetTPPLeadingDim::run(snippets::lowered::LinearIR& linear_ir,
                           snippets::lowered::LinearIR::constExprIt begin,
                           snippets::lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SetTPPLeadingDim")
    if (linear_ir.empty()) {
        return false;
    }

    bool modified = false;
    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& expr = *expr_it;
        const auto& node = expr->get_node();
        auto tpp_expr = std::dynamic_pointer_cast<modifier::TensorProcessingPrimitive>(node);
        if (!tpp_expr) {
            continue;
        }

        OPENVINO_ASSERT(tpp_expr->is_full_memory_access_op(node), "TPP Op is expected to be MemoryAccess on all ports");

        for (size_t i = 0; i < expr->get_input_count(); i++) {
            const auto ld = get_leading_dim(expr->get_input_port(i), linear_ir.get_loop_manager());
            tpp_expr->set_input_stride(ld, i);
        }
        for (size_t i = 0; i < expr->get_output_count(); i++) {
            const auto ld = get_leading_dim(expr->get_output_port(i), linear_ir.get_loop_manager());
            tpp_expr->set_output_stride(ld, i);
        }
        modified = true;
    }

    return modified;
}

}  // namespace ov::intel_cpu::tpp::pass
