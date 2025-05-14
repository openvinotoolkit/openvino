// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_cpu_blocking.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

namespace ov::intel_cpu::pass {
using LinearIR = snippets::lowered::LinearIR;
using LoopPort = snippets::lowered::LoopPort;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;
using namespace ov::intel_cpu::brgemm_utils;
using namespace ov::snippets::lowered;
using namespace ov::snippets::utils;

bool GemmCPUBlocking::DummyPass::run([[maybe_unused]] LinearIR& linear_ir,
                                     [[maybe_unused]] LinearIR::constExprIt begin,
                                     [[maybe_unused]] LinearIR::constExprIt end) {
    return true;
}
std::shared_ptr<snippets::lowered::pass::PassBase> GemmCPUBlocking::DummyPass::merge(
    const std::shared_ptr<snippets::lowered::pass::PassBase>& other) {
    return !other || ov::is_type<DummyPass>(other) ? std::make_shared<DummyPass>() : nullptr;
}

std::tuple<size_t, size_t, size_t> GemmCPUBlocking::get_blocking_params(
    const ov::snippets::lowered::ExpressionPtr& brgemm_expr) const {
    const auto brgemm = ov::as_type_ptr<ov::intel_cpu::aarch64::GemmCPU>(brgemm_expr->get_node());
    assert(brgemm && "GemmCPU is expected!");

    size_t m_blk, n_blk, k_blk;
    std::tie(m_blk, n_blk, k_blk) = BrgemmBlockingBase::get_blocking_params(brgemm_expr);
    return std::make_tuple(m_blk, n_blk, k_blk);
}

SpecificIterationHandlers GemmCPUBlocking::get_k_loop_handlers(size_t work_amount, size_t block_size) const {
    SpecificIterationHandlers handlers =
        ov::snippets::lowered::pass::BrgemmBlockingBase::get_k_loop_handlers(work_amount, block_size);
    handlers.register_pass<SpecificLoopIterType::FIRST_ITER, DummyPass>();
    return handlers;
}

bool GemmCPUBlocking::mark_blocking_loops(LinearIR& linear_ir,
                                          const LinearIR::constExprIt& brgemm_it,
                                          size_t m_block,
                                          size_t n_block,
                                          size_t k_block) {
    const auto& brgemm_expr = *brgemm_it;
    const auto brgemm = ov::as_type_ptr<ov::intel_cpu::aarch64::GemmCPU>(brgemm_expr->get_node());

    auto res = ov::snippets::lowered::pass::BrgemmBlockingBase::mark_blocking_loops(linear_ir,
                                                                                    brgemm_it,
                                                                                    m_block,
                                                                                    n_block,
                                                                                    k_block);
    return res;
}
}  // namespace ov::intel_cpu::pass
