// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
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
 * @brief Get inner n block that is required by KleidiAI
 * @return Inner n block size
 */
size_t get_inner_n_block(const ov::element::Type& precision);

/**
 * @brief Get k padding size that is required by KleidiAI
 * @return k padding size
 */
size_t get_k_pad_size(const ov::element::Type& precision);

/**
 * @brief Get the offset in bytes to the packed RHS data for the specified N index and K dimension.
 * @return Packed RHS offset in bytes
 */
size_t get_rhs_packed_offset(const ov::element::Type& precision, size_t n_idx, size_t K);

/**
 * @brief Get the size in bytes of the packed RHS buffer.
 * @return Packed RHS size in bytes
 */
size_t get_rhs_packed_size(const ov::element::Type& precision, size_t N, size_t K);

/**
 * @brief Get the N step required by the packed RHS layout.
 * @return N step
 */
size_t get_rhs_packed_n_step(const ov::element::Type& precision);

}  // namespace ov::intel_cpu::aarch64::gemm_utils::repacking
