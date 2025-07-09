// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>

#include "openvino/core/dimension.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu::aarch64::gemm_utils::repacking {
/**
 * @brief Retrieves the expression pointer for the gemm_copy_b expression corresponding to the given GemmCPU
 * expression.
 * @param gemm_expr The expression pointer for the GemmCPU operation.
 * @return The expression pointer for the GemmCopyB operation.
 */
snippets::lowered::ExpressionPtr get_copy_b_expr(const snippets::lowered::ExpressionPtr& gemm_expr);

/**
 * @brief Retrieves the expression pointers for the gemm expressions corresponding to the given gemm_copy_b
 * expression.
 * @param gemm_expr The expression pointer for the gemm_copy_b operation.
 * @return The expression pointers for the gemm operation.
 */
std::vector<snippets::lowered::ExpressionPtr> get_gemm_exprs(const snippets::lowered::ExpressionPtr& gemm_copyb_expr);

/**
 * @brief Get inner n block that is required by KleidiAI
 * @return Inner n block size
 */
size_t get_inner_n_block(const ov::element::Type& precision);

/**
 * @brief Get k padding size that is required by KleidiAI
 * @return k padding size
 */
size_t get_k_pad_size(const ov::element::Type& precision);

}  // namespace ov::intel_cpu::aarch64::gemm_utils::repacking
