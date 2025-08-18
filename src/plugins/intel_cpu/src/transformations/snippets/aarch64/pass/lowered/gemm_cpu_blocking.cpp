// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_cpu_blocking.hpp"

#include <cassert>
#include <cstddef>
#include <memory>
#include <tuple>

#include "openvino/core/type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/brgemm_blocking.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/lowered/specific_loop_iter_handlers.hpp"
#include "snippets/lowered/specific_loop_iter_types.hpp"
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

bool GemmCPUBlocking::DummyPass::run([[maybe_unused]] snippets::lowered::LinearIR& linear_ir,
                                     [[maybe_unused]] snippets::lowered::LinearIR::constExprIt begin,
                                     [[maybe_unused]] snippets::lowered::LinearIR::constExprIt end) {
    return true;
}

std::shared_ptr<snippets::lowered::pass::PassBase> GemmCPUBlocking::DummyPass::merge(
    const std::shared_ptr<snippets::lowered::pass::PassBase>& other) {
    return !other || ov::is_type<DummyPass>(other) ? std::make_shared<DummyPass>() : nullptr;
}

snippets::lowered::SpecificIterationHandlers GemmCPUBlocking::get_k_loop_handlers(size_t work_amount,
                                                                                  size_t block_size) const {
    snippets::lowered::SpecificIterationHandlers handlers =
        ov::snippets::lowered::pass::BrgemmBlockingBase::get_default_blocking_loop_handlers(work_amount, block_size);
    handlers.register_pass<ov::snippets::lowered::SpecificLoopIterType::FIRST_ITER, DummyPass>();
    return handlers;
}

}  // namespace ov::intel_cpu::pass
