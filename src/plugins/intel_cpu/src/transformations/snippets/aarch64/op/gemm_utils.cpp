// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_utils.hpp"

#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/op/buffer.hpp"
#include "transformations/snippets/aarch64/op/gemm_copy_b.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"
#include "utils/general_utils.h"

using namespace ov::snippets::utils;

namespace ov {
namespace intel_cpu::aarch64::gemm_utils {
namespace repacking {
ov::snippets::lowered::ExpressionPtr get_copy_b_expr(const ov::snippets::lowered::ExpressionPtr& gemm_expr) {
    OPENVINO_ASSERT(ov::is_type<GemmCPU>(gemm_expr->get_node()),
                    "get_copy_b_expr must be called only for GemmCPU node");
    auto b_input_expr = gemm_expr->get_input_port_connector(1)->get_source().get_expr();
    if (ov::is_type<GemmCopyB>(b_input_expr->get_node())) {
        return b_input_expr;
    }
    if (ov::is_type<snippets::lowered::BufferExpression>(b_input_expr)) {
        OPENVINO_ASSERT(b_input_expr->get_input_count() >= 1,
                        "BufferExpression on gemm's B input must have at least one input");
        auto input_buffer_expr = b_input_expr->get_input_port_connector(0)->get_source().get_expr();
        if (ov::is_type<GemmCopyB>(input_buffer_expr->get_node())) {
            return input_buffer_expr;
        }
    }
    return nullptr;
}
}  // namespace repacking
}  // namespace intel_cpu::aarch64::gemm_utils
}  // namespace ov
