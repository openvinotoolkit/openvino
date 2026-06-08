// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_utils.hpp"

#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p32x1b_x16_x16_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x32p16x1b_x32_x32_neon.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "transformations/snippets/aarch64/op/gemm_copy_b.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"
#include "transformations/snippets/aarch64/pass/lowered/expressions/gemm_copy_b_buffer_expressions.hpp"

namespace ov::intel_cpu::aarch64::gemm_utils::repacking {
size_t get_rhs_packed_size(size_t N, size_t K, const ov::element::Type& precision) {
    if (precision == ov::element::f16) {
        return kai_get_rhs_packed_size_rhs_pack_kxn_x16p32x1b_x16_x16_neon(N, K);
    }
    if (precision == ov::element::f32) {
        return kai_get_rhs_packed_size_rhs_pack_kxn_x32p16x1b_x32_x32_neon(N, K);
    }
    OPENVINO_THROW("Unsupported precision for aarch64 GEMM weights repacking: ", precision.get_type_name());
}

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
