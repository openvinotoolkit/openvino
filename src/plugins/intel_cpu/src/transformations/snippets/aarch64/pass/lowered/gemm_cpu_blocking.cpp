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

// move to common and reused for x64 and arm. ifdefine to use gemm_cpu or brgemm_cpu
bool GemmCPUBlocking::DummyPass::run([[maybe_unused]] LinearIR& linear_ir,
                                     [[maybe_unused]] LinearIR::constExprIt begin,
                                     [[maybe_unused]] LinearIR::constExprIt end) {
    return true;
}
std::shared_ptr<snippets::lowered::pass::PassBase> GemmCPUBlocking::DummyPass::merge(
    const std::shared_ptr<snippets::lowered::pass::PassBase>& other) {
    return !other || ov::is_type<DummyPass>(other) ? std::make_shared<DummyPass>() : nullptr;
}

LinearIR::constExprIt GemmCPUBlocking::move_new_memory_buffer(LinearIR& linear_ir,
                                                              const LinearIR::constExprIt& brgemm_it) {
    const auto& brgemm_expr = brgemm_it->get();
    const auto wsp_expr = brgemm_expr->get_input_port_connector(2)->get_source().get_expr();
    const auto wsp_buffer = ov::as_type_ptr<ov::snippets::lowered::BufferExpression>(wsp_expr);
    OPENVINO_ASSERT(wsp_buffer && wsp_buffer->is_independent_memory(), "Incorrect Scratchpad buffer for Brgemm AMX");
    // If scratchpad with temp memory is not explicitly before Brgemm, need to move to there.
    if (wsp_expr != *std::prev(brgemm_it)) {
        const auto wsp_it = linear_ir.find(wsp_expr);
        linear_ir.move(wsp_it, brgemm_it);
    }
    return std::prev(brgemm_it);
}

// size_t GemmCPUBlocking::get_default_n_blk([[maybe_unused]] size_t n) const {
//     return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ? 64 : 24;
// }

std::tuple<size_t, size_t, size_t> GemmCPUBlocking::get_blocking_params(
    const ov::snippets::lowered::ExpressionPtr& brgemm_expr) const {
    const auto brgemm = ov::as_type_ptr<ov::intel_cpu::aarch64::GemmCPU>(brgemm_expr->get_node());
    assert(brgemm && "GemmCPU is expected!");

    size_t m_blk, n_blk, k_blk;
    std::tie(m_blk, n_blk, k_blk) = BrgemmBlockingBase::get_blocking_params(brgemm_expr);
    std::cout << "m_blk:" << m_blk << std::endl;
    std::cout << "n_blk:" << n_blk << std::endl;
    std::cout << "k_blk:" << k_blk << std::endl;
    // [TODO]: K,N blocking is functionally enabled, need to turn it on after blocking heuristic is updated to cover
    //         the low precision cases (ticket: 156014)
    //         Please note that FP32 MatMul with `transposed_b=true` has type `with_repacking` despite the precision.
    // const auto precision = brgemm_expr->get_node()->get_input_element_type(1);
    // if (with_repacking(brgemm->get_type()) && precision != element::f32) {
    //     n_blk = get_full_dim_value();
    //     k_blk = get_full_dim_value();
    // }
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
