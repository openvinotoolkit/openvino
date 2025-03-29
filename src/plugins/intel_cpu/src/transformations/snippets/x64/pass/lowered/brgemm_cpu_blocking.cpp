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

namespace ov::intel_cpu::pass {
using LinearIR = snippets::lowered::LinearIR;
using LoopPort = snippets::lowered::LoopPort;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;
using namespace ov::intel_cpu::brgemm_utils;
using namespace ov::snippets::lowered;
using namespace ov::snippets::utils;

bool BrgemmCPUBlocking::DummyPass::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    return true;
}
std::shared_ptr<snippets::lowered::pass::PassBase> BrgemmCPUBlocking::DummyPass::merge(
    const std::shared_ptr<snippets::lowered::pass::PassBase>& other) {
    return !other || ov::is_type<DummyPass>(other) ? std::make_shared<DummyPass>() : nullptr;
}

LinearIR::constExprIt BrgemmCPUBlocking::move_new_memory_buffer(LinearIR& linear_ir,
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

size_t BrgemmCPUBlocking::get_default_n_blk(size_t n) const {
    return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ? 64 : 24;
}

std::tuple<size_t, size_t, size_t> BrgemmCPUBlocking::get_blocking_params(
    const ov::snippets::lowered::ExpressionPtr& brgemm_expr) const {
    const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(brgemm_expr->get_node());
    OPENVINO_ASSERT(brgemm, "BrgemmCPU is expected!");

    size_t m_blk, n_blk, k_blk;
    std::tie(m_blk, n_blk, k_blk) = BrgemmBlockingBase::get_blocking_params(brgemm_expr);
    // [TODO]: K,N blocking is functionally enabled, need to turn it on after blocking heuristic is updated to cover
    //         the low precision cases (ticket: 156014)
    //         Please note that FP32 MatMul with `transposed_b=true` has type `with_repacking` despite the precision.
    const auto precision = brgemm_expr->get_node()->get_input_element_type(1);
    if (with_repacking(brgemm->get_type()) && precision != element::f32) {
        n_blk = get_full_dim_value();
        k_blk = get_full_dim_value();
    }
    return std::make_tuple(m_blk, n_blk, k_blk);
}

SpecificIterationHandlers BrgemmCPUBlocking::get_k_loop_handlers(size_t work_amount, size_t block_size) const {
    SpecificIterationHandlers handlers =
        ov::snippets::lowered::pass::BrgemmBlockingBase::get_k_loop_handlers(work_amount, block_size);
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

    auto res = ov::snippets::lowered::pass::BrgemmBlockingBase::mark_blocking_loops(linear_ir,
                                                                                    brgemm_it,
                                                                                    m_block,
                                                                                    n_block,
                                                                                    k_block);

    if (stand_alone(type)) {
        return res;
    }

    const auto copy_b_expr = repacking::get_copy_b_expr(brgemm_expr);
    if (copy_b_expr) {
        const ov::snippets::VectorDims full_subtensor(2, get_full_dim_value());
        copy_b_expr->get_input_port_descriptor(0)->set_subtensor(full_subtensor);
        copy_b_expr->get_output_port_descriptor(0)->set_subtensor(full_subtensor);
    }
    if (with_amx(type)) {
        move_new_memory_buffer(linear_ir, brgemm_it);
        auto buffer_it = std::prev(brgemm_it);
        buffer_it->get()->set_loop_ids(brgemm_expr->get_loop_ids());
    }

    const auto& loop_manager = linear_ir.get_loop_manager();
    if (with_compensations(type)) {
        const ov::snippets::VectorDims compensations_subtensor{1, get_full_dim_value()};
        OPENVINO_ASSERT(brgemm_expr->get_input_count() == 3, "Brgemm must have 3 inputs in case of compensations.");
        OPENVINO_ASSERT(copy_b_expr, "BrgemmCopyB must be present in case of compensations.");
        const auto& compens_port = brgemm_expr->get_input_port(2);
        compens_port.get_descriptor_ptr()->set_subtensor(compensations_subtensor);
        copy_b_expr->get_output_port_descriptor(1)->set_subtensor(compensations_subtensor);

        const auto& loop_ids = brgemm_expr->get_loop_ids();
        size_t i = 0;
        LoopInfoPtr loop_info = nullptr;
        auto update_loop_info = [&](LoopPort&& new_port) {
            OPENVINO_ASSERT(i < loop_ids.size(), "Attempt to access invalid loop id");
            loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(loop_ids[i++]);
            const auto& in_ports = loop_info->get_input_ports();
            OPENVINO_ASSERT(in_ports.size() > 1, "Invalid number of input loop ports");
            loop_info->replace_with_new_ports(in_ports[1], {in_ports[1], new_port});
        };
        if (!is_full_dim_value(m_block)) {
            update_loop_info(LoopPort::create<LoopPort::Type::NotProcessed>(compens_port));
        }

        if (!is_full_dim_value(n_block)) {
            update_loop_info(LoopPort::create<LoopPort::Type::Incremented>(compens_port, 0));
        }

        if (!is_full_dim_value(k_block)) {
            update_loop_info(LoopPort::create<LoopPort::Type::NotIncremented>(compens_port, 1));
        }
    }
    return true;
}
}  // namespace ov::intel_cpu::pass
