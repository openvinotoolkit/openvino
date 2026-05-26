// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_utils.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "snippets/lowered/expression.hpp"
#include "transformations/snippets/aarch64/op/gemm_copy_b.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"
#include "transformations/snippets/aarch64/pass/lowered/expressions/gemm_copy_b_buffer_expressions.hpp"

namespace ov::intel_cpu::aarch64::gemm_utils::repacking {
ov::snippets::lowered::ExpressionPtr get_copy_b_expr(const ov::snippets::lowered::ExpressionPtr& gemm_expr) {
    OPENVINO_ASSERT(ov::is_type<GemmCPU>(gemm_expr->get_node()),
                    "get_copy_b_expr must be called only for GemmCPU node");
    auto b_input_expr = gemm_expr->get_input_expr_ptr(1);
    if (ov::is_type<GemmCopyB>(b_input_expr->get_node())) {
        return b_input_expr;
    }
    if (ov::is_type<RepackedWeightsBufferExpression>(b_input_expr)) {
        OPENVINO_ASSERT(b_input_expr->get_input_count() == 1,
                        "RepackedWeightsBufferExpression on gemm's B input must has one input");
        auto input_buffer_expr = b_input_expr->get_input_expr_ptr(0);
        if (ov::is_type<GemmCopyB>(input_buffer_expr->get_node())) {
            return input_buffer_expr;
        }
    }
    return nullptr;
}

}  // namespace ov::intel_cpu::aarch64::gemm_utils::repacking
