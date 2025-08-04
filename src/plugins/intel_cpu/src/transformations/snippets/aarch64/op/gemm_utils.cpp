// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_utils.hpp"

#include <cstddef>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "transformations/snippets/aarch64/op/gemm_copy_b.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"
#include "transformations/snippets/aarch64/pass/lowered/expressions/gemm_copy_b_buffer_expressions.hpp"

using namespace ov::snippets::utils;

namespace ov::intel_cpu::aarch64::gemm_utils::repacking {
ov::snippets::lowered::ExpressionPtr get_copy_b_expr(const ov::snippets::lowered::ExpressionPtr& gemm_expr) {
    OPENVINO_ASSERT(ov::is_type<GemmCPU>(gemm_expr->get_node()),
                    "get_copy_b_expr must be called only for GemmCPU node");
    auto b_input_expr = gemm_expr->get_input_port_connector(1)->get_source().get_expr();
    if (ov::is_type<GemmCopyB>(b_input_expr->get_node())) {
        return b_input_expr;
    }
    if (ov::is_type<RepackedWeightsBufferExpression>(b_input_expr)) {
        OPENVINO_ASSERT(b_input_expr->get_input_count() == 1,
                        "RepackedWeightsBufferExpression on gemm's B input must has one input");
        auto input_buffer_expr = b_input_expr->get_input_port_connector(0)->get_source().get_expr();
        if (ov::is_type<GemmCopyB>(input_buffer_expr->get_node())) {
            return input_buffer_expr;
        }
    }
    return nullptr;
}

std::vector<snippets::lowered::ExpressionPtr> get_gemm_exprs(
    const ov::snippets::lowered::ExpressionPtr& gemm_copyb_expr) {
    OPENVINO_ASSERT(ov::is_type<GemmCopyB>(gemm_copyb_expr->get_node()),
                    "get_gemm_exprs must be called only for GemmCopyB node");
    OPENVINO_ASSERT(gemm_copyb_expr->get_output_count() == 1, "gemm copyb expr must has one output");
    std::vector<snippets::lowered::ExpressionPtr> result;
    auto copyb_output_expr = gemm_copyb_expr->get_output_port_connector(0)->get_consumers().begin()->get_expr();
    if (ov::is_type<GemmCPU>(copyb_output_expr->get_node())) {
        result.push_back(copyb_output_expr);
    }
    if (ov::is_type<RepackedWeightsBufferExpression>(copyb_output_expr)) {
        OPENVINO_ASSERT(copyb_output_expr->get_output_count() == 1, "gemm copyb buffer expr must has one output");
        // repacked buffer could connect gemm expr in main loop and tail loop.
        const auto& consumers = copyb_output_expr->get_output_port_connector(0)->get_consumers();
        for (const auto& consumer : consumers) {
            if (ov::is_type<GemmCPU>(consumer.get_expr()->get_node())) {
                result.push_back(consumer.get_expr());
            }
        }
    }
    return result;
}

size_t get_inner_n_block(const ov::element::Type& precision) {
    OPENVINO_ASSERT(precision == element::f32, "Only f32 is supported for snippets Matmul");
    return 8;
}

size_t get_k_pad_size(const ov::element::Type& precision) {
    OPENVINO_ASSERT(precision == element::f32, "Only f32 is supported for snippets Matmul");
    return 1;
}

}  // namespace ov::intel_cpu::aarch64::gemm_utils::repacking
