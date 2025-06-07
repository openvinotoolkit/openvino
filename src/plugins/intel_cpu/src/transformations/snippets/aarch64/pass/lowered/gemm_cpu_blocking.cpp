// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_cpu_blocking.hpp"

#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu::pass {

std::tuple<size_t, size_t, size_t> GemmCPUBlocking::get_blocking_params(
    const ov::snippets::lowered::ExpressionPtr& gemm_expr) const {
    const auto gemm = ov::as_type_ptr<ov::intel_cpu::aarch64::GemmCPU>(gemm_expr->get_node());
    assert(gemm && "GemmCPU is expected!");

    const auto [m, n, k] = get_brgemm_dimensions(gemm_expr);

    const auto default_m_blk = GemmCPUBlocking::get_default_m_blk();
    const auto default_n_blk = GemmCPUBlocking::get_default_n_blk();

    size_t m_blk = get_corrected_blk_size_by_dim(m, default_m_blk);
    size_t n_blk = get_corrected_blk_size_by_dim(n, default_n_blk);
    size_t k_blk = GemmCPUBlocking::get_default_k_blk();

    return std::make_tuple(m_blk, n_blk, k_blk);
}

size_t GemmCPUBlocking::get_default_m_blk() {
    return 32;
}

size_t GemmCPUBlocking::get_default_n_blk() {
    return 64;
}

size_t GemmCPUBlocking::get_default_k_blk() {
    return ov::snippets::utils::get_full_dim_value();
}

}  // namespace ov::intel_cpu::pass
