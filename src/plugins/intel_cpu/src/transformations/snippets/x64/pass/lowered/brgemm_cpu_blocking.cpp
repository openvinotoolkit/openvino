// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_cpu_blocking.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"


namespace ov {
namespace intel_cpu {
namespace pass {
using LinearIR = snippets::lowered::LinearIR;
using LoopPort = snippets::lowered::LoopPort;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;
using namespace ov::intel_cpu::brgemm_utils;
using namespace ov::snippets::lowered;
using namespace ov::snippets::utils;

bool BrgemmCPUBlocking::DummyPass::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    return true;
}
std::shared_ptr<snippets::lowered::pass::PassBase> BrgemmCPUBlocking::DummyPass::merge(const std::shared_ptr<snippets::lowered::pass::PassBase>& other) {
    return !other || ov::is_type<DummyPass>(other) ? std::make_shared<DummyPass>() : nullptr;
}

LinearIR::constExprIt BrgemmCPUBlocking::move_new_memory_buffer(LinearIR& linear_ir, const LinearIR::constExprIt& brgemm_it) {
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

LinearIR::constExprIt BrgemmCPUBlocking::get_loop_begin_pos(LinearIR& linear_ir, const LinearIR::constExprIt& brgemm_it, const ExpressionPtr& copy_b_expr) {
    auto loop_begin_it = brgemm_it;
    const auto& brgemm_expr = *brgemm_it;
    const auto brgemm = ov::as_type_ptr<intel_cpu::BrgemmCPU>(brgemm_expr->get_node());
    OPENVINO_ASSERT(brgemm, "get_loop_begin_pos must be called only for BrgemmCPU expression");
    if (with_amx(brgemm->get_type()))
        loop_begin_it = move_new_memory_buffer(linear_ir, brgemm_it);
    if (copy_b_expr)
        loop_begin_it = linear_ir.find(copy_b_expr);
    return loop_begin_it;
}

size_t BrgemmCPUBlocking::get_default_n_blk(size_t n) const {
    return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ? 64 : 24;
}

std::tuple<size_t, size_t, size_t> BrgemmCPUBlocking::get_blocking_params(const ov::snippets::lowered::ExpressionPtr& brgemm_expr) const {
    const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(brgemm_expr->get_node());
    OPENVINO_ASSERT(brgemm, "BrgemmCPU is expected!");

    size_t m_blk, n_blk, k_blk;
    std::tie(m_blk, n_blk, k_blk) = BrgemmBlockingBase::get_blocking_params(brgemm_expr);
    if (with_repacking(brgemm->get_type())) {
        n_blk = get_full_dim_value();
        k_blk = get_full_dim_value();
    }
    return std::make_tuple(m_blk, n_blk, k_blk);
}

SpecificIterationHandlers BrgemmCPUBlocking::get_k_loop_handlers(size_t work_amount, size_t block_size) const {
    SpecificIterationHandlers handlers = ov::snippets::lowered::pass::BrgemmBlockingBase::get_k_loop_handlers(work_amount, block_size);
    handlers.register_pass<SpecificLoopIterType::FIRST_ITER, DummyPass>();
    return handlers;
}

bool BrgemmCPUBlocking::mark_blocking_loops(LinearIR& linear_ir,
                                            const LinearIR::constExprIt& brgemm_it,
                                            size_t m_block,
                                            size_t n_block,
                                            size_t k_block) {
    const auto& brgemm_expr = *brgemm_it;
    const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(brgemm_expr->get_node());
    const auto type = brgemm->get_type();

    if (stand_alone(type))
        return ov::snippets::lowered::pass::BrgemmBlockingBase::mark_blocking_loops(linear_ir, brgemm_it, m_block, n_block, k_block);

    brgemm_expr->get_input_port_descriptor(0)->set_subtensor({m_block, k_block});
    brgemm_expr->get_input_port_descriptor(1)->set_subtensor({k_block, n_block});
    brgemm_expr->get_output_port_descriptor(0)->set_subtensor({m_block, n_block});

    const auto copy_b_expr = linear_ir.get_expr_by_node(brgemm->get_brgemm_copy());
    copy_b_expr->get_input_port_descriptor(0)->set_subtensor({k_block, n_block});
    copy_b_expr->get_output_port_descriptor(0)->set_subtensor({k_block, n_block});
    if (with_compensations(type)) {
        const ov::snippets::VectorDims compensations_subtensor{1, n_block};
        OPENVINO_ASSERT(brgemm_expr->get_input_count() == 3, "Brgemm must have 3 inputs in case of compensations.");
        brgemm_expr->get_input_port_descriptor(2)->set_subtensor(compensations_subtensor);
        copy_b_expr->get_output_port_descriptor(1)->set_subtensor(compensations_subtensor);
    }

    const auto& loop_manager = linear_ir.get_loop_manager();
    if (!is_full_dim_value(k_block)) {
        const auto loop_begin = get_loop_begin_pos(linear_ir, brgemm_it, copy_b_expr);
        const std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), true, 0),
                                            LoopPort(copy_b_expr->get_input_port(0), true, 1)};
        const std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), false)};
        mark_k_blocking(loop_manager, loop_begin, std::next(brgemm_it), entries, exits, k_block);
    }
    if (!is_full_dim_value(n_block)) {
        const auto loop_begin = get_loop_begin_pos(linear_ir, brgemm_it, copy_b_expr);
        const std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), false),
                                            LoopPort(copy_b_expr->get_input_port(0), true)};
        const std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
        mark_n_blocking(loop_manager, loop_begin, std::next(brgemm_it), entries, exits, n_block);
    }
    if (!is_full_dim_value(m_block)) {
        const bool include_repacking = !is_full_dim_value(k_block) || !is_full_dim_value(n_block);
        const auto loop_begin = get_loop_begin_pos(linear_ir, brgemm_it, include_repacking ? copy_b_expr : nullptr);
        const auto b_input_port = include_repacking ? copy_b_expr->get_input_port(0) : brgemm_expr->get_input_port(1);
        std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), true), LoopPort(b_input_port, false)};
        if (!include_repacking && with_compensations(type))
            entries.emplace_back(brgemm_expr->get_input_port(2), false);
        const std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
        mark_m_blocking(loop_manager, loop_begin, std::next(brgemm_it), entries, exits, m_block);
    }
    return true;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov