// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_cpu_blocking.hpp"

#include <cassert>
#include <cstddef>
#include <tuple>

#include "openvino/core/type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"

namespace ov::intel_cpu::pass {

std::tuple<size_t, size_t, size_t> GemmCPUBlocking::get_blocking_params(
    const ov::snippets::lowered::ExpressionPtr& gemm_expr) const {
    const auto gemm = ov::as_type_ptr<ov::intel_cpu::aarch64::GemmCPU>(gemm_expr->get_node());
    assert(gemm && "GemmCPU is expected!");

    const auto [m, n, k] = get_brgemm_dimensions(gemm_expr);

    const size_t& default_m_blk = 32;
    const size_t& default_n_blk = 64;

    const size_t& m_blk = get_corrected_blk_size_by_dim(m, default_m_blk);
    const size_t& n_blk = get_corrected_blk_size_by_dim(n, default_n_blk);
    const size_t& k_blk = ov::snippets::utils::get_full_dim_value();

    return std::make_tuple(m_blk, n_blk, k_blk);
}

}  // namespace ov::intel_cpu::pass
