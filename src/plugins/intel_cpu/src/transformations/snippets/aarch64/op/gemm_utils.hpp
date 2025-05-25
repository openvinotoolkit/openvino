// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>

#include "openvino/core/dimension.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace intel_cpu::aarch64::gemm_utils {
namespace repacking {
/**
 * @brief Retrieves the expression pointer for the gemm_copy_b expression corresponding to the given GemmCPU
 * expression.
 * @param gemm_expr The expression pointer for the GemmCPU operation.
 * @return The expression pointer for the GemmCopyB operation.
 */
snippets::lowered::ExpressionPtr get_copy_b_expr(const snippets::lowered::ExpressionPtr& gemm_expr);
}  // namespace repacking
}  // namespace intel_cpu::aarch64::gemm_utils
}  // namespace ov
