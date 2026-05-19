// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/expression.hpp"

namespace ov::intel_cpu::aarch64::gemm_utils::repacking {
/**
 * @brief Retrieves the expression pointer for the gemm_copy_b expression corresponding to the given GemmCPU
 * expression.
 * @param gemm_expr The expression pointer for the GemmCPU operation.
 * @return The expression pointer for the GemmCopyB operation.
 */
snippets::lowered::ExpressionPtr get_copy_b_expr(const snippets::lowered::ExpressionPtr& gemm_expr);

}  // namespace ov::intel_cpu::aarch64::gemm_utils::repacking
