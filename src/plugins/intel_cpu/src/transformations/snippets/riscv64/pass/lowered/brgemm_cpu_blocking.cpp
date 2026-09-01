// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_cpu_blocking.hpp"

#include <cassert>
#include <cstddef>
#include <tuple>

#include "snippets/lowered/expression.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu::pass {

std::tuple<size_t, size_t, size_t> BrgemmCPUBlocking::get_blocking_params(
    const ov::snippets::lowered::ExpressionPtr& brgemm_expr) const {
    assert(ov::is_type<ov::intel_cpu::BrgemmCPU>(brgemm_expr->get_node()) && "BrgemmCPU is expected");

    const auto m = std::get<0>(get_brgemm_dimensions(brgemm_expr));
    constexpr size_t default_m_block = 32;
    return {get_corrected_blk_size_by_dim(m, default_m_block),
            ov::snippets::utils::get_full_dim_value(),
            ov::snippets::utils::get_full_dim_value()};
}

}  // namespace ov::intel_cpu::pass
