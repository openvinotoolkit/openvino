// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_cpu_blocking.hpp"

#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_port.hpp"
#include "snippets/lowered/pass/brgemm_blocking.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/lowered/specific_loop_iter_handlers.hpp"
#include "snippets/lowered/specific_loop_iter_types.hpp"
#include "snippets/shape_types.hpp"
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

bool BrgemmCPUBlocking::DummyPass::run([[maybe_unused]] LinearIR& linear_ir,
                                       [[maybe_unused]] LinearIR::constExprIt begin,
                                       [[maybe_unused]] LinearIR::constExprIt end) {
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

void BrgemmCPUBlocking::update_loop_infos(
    const ov::snippets::lowered::LoopManagerPtr& loop_manager,
    const std::vector<size_t>& loop_ids,
    const std::vector<std::pair<size_t, std::vector<LoopPort>>>& block_to_new_ports) {
    size_t i = 0;
    for (const auto& pair : block_to_new_ports) {
        if (is_full_dim_value(pair.first)) {
            continue;
        }

        const auto& new_ports = pair.second;
        OPENVINO_ASSERT(i < loop_ids.size(), "Attempt to access invalid loop id");
        const auto loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(loop_ids[i++]);
        const auto& in_ports = loop_info->get_input_ports();
        OPENVINO_ASSERT(in_ports.size() > 1, "Invalid number of input loop ports");
        std::vector<LoopPort> replacement_ports{in_ports.back()};
        replacement_ports.insert(replacement_ports.end(), new_ports.begin(), new_ports.end());
        loop_info->replace_with_new_ports(in_ports.back(), replacement_ports);
    }
}

void BrgemmCPUBlocking::create_not_processed_postops_ports(const ov::snippets::lowered::ExpressionPtr& brgemm_expr,
                                                           const ov::snippets::lowered::LoopManagerPtr& loop_manager,
                                                           size_t m_block,
                                                           size_t n_block,
                                                           size_t k_block) {
    const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(brgemm_expr->get_node());
    OPENVINO_ASSERT(brgemm, "BrgemmCPU is expected!");
    const auto postops_inputs = brgemm->get_postop_inputs();

    std::vector<LoopPort> new_ports;
    const auto gemm_inputs_count = brgemm->get_gemm_inputs_count();
    for (size_t i = gemm_inputs_count; i < gemm_inputs_count + postops_inputs.size(); ++i) {
        const auto& postop_input_port = brgemm_expr->get_input_port(i);
        postop_input_port.get_descriptor_ptr()->set_subtensor({get_full_dim_value(), get_full_dim_value()});
        new_ports.push_back(LoopPort::create<LoopPort::Type::NotProcessed>(postop_input_port));
    }
    update_loop_infos(loop_manager,
                      brgemm_expr->get_loop_ids(),
                      {{m_block, new_ports}, {n_block, new_ports}, {k_block, new_ports}});
}

bool BrgemmCPUBlocking::is_kn_blocking_supported(const ov::element::Type& input_type) {
    return input_type == element::f32;
}

std::tuple<size_t, size_t, size_t> BrgemmCPUBlocking::get_blocking_params(
    const ov::snippets::lowered::ExpressionPtr& brgemm_expr) const {
    const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(brgemm_expr->get_node());
    assert(brgemm && "BrgemmCPU is expected!");
    const auto& brgemm_config = brgemm->get_config();

    const auto [m, n, k] = get_brgemm_dimensions(brgemm_expr);

    const auto default_m_blk = 32;
    const auto default_n_blk = 64;
    const auto default_k_blk = !ov::snippets::utils::is_dynamic_value(k) && k > 1024 ? 1024 : 512;

    size_t m_blk = get_corrected_blk_size_by_dim(m, default_m_blk);
    size_t n_blk =
        get_corrected_blk_size_by_dim(n, brgemm_config.are_wei_blocked() ? brgemm_config.wei_n_blk() : default_n_blk);
    size_t k_blk = get_corrected_blk_size_by_dim(k, default_k_blk);

    // [TODO]: K,N blocking is functionally enabled, need to turn it on after blocking heuristic is updated to cover
    //         the low precision cases (ticket: 156014)
    if (is_kn_blocking_supported(brgemm->get_input_element_type(1))) {
        OPENVINO_ASSERT(brgemm->get_postops_config().post_ops.len() == 0,
                        "Blocking for Brgemm with postops is not supported");
    } else {
        OPENVINO_ASSERT(!brgemm_config.are_wei_blocked(),
                        "Weights of Brgemm cannot be repacked in blocked format if KN blocking is not supported");
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
    const auto res = ov::snippets::lowered::pass::BrgemmBlockingBase::mark_blocking_loops(linear_ir,
                                                                                          brgemm_it,
                                                                                          m_block,
                                                                                          n_block,
                                                                                          k_block);
    const auto& brgemm_expr = *brgemm_it;
    const auto& brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(brgemm_expr->get_node());
    const auto& brgemm_config = brgemm->get_config();
    const auto& loop_manager = linear_ir.get_loop_manager();
    if (brgemm_config.with_wei_repacking()) {
        const auto copy_b_expr = repacking::get_copy_b_expr(brgemm_expr);
        if (copy_b_expr) {
            const ov::snippets::VectorDims full_subtensor(2, get_full_dim_value());
            copy_b_expr->get_input_port_descriptor(0)->set_subtensor(full_subtensor);
            copy_b_expr->get_output_port_descriptor(0)->set_subtensor(full_subtensor);
        }
        if (brgemm_config.is_amx()) {
            move_new_memory_buffer(linear_ir, brgemm_it);
            OPENVINO_ASSERT(brgemm_it != linear_ir.begin(), "Brgemm must have buffer before itself");
            auto buffer_it = std::prev(brgemm_it);
            buffer_it->get()->set_loop_ids(brgemm_expr->get_loop_ids());
        }

        if (brgemm_config.with_compensations()) {
            const ov::snippets::VectorDims compensations_subtensor{1, get_full_dim_value()};
            OPENVINO_ASSERT(brgemm_expr->get_input_count() >= 3,
                            "Brgemm must have at least 3 inputs in case of compensations.");
            OPENVINO_ASSERT(copy_b_expr, "BrgemmCopyB must be present in case of compensations.");
            const auto& compensations_port = brgemm_expr->get_input_port(2);
            compensations_port.get_descriptor_ptr()->set_subtensor(compensations_subtensor);
            copy_b_expr->get_output_port_descriptor(1)->set_subtensor(compensations_subtensor);
            update_loop_infos(loop_manager,
                              brgemm_expr->get_loop_ids(),
                              {{m_block, {LoopPort::create<LoopPort::Type::NotProcessed>(compensations_port)}},
                               {n_block, {LoopPort::create<LoopPort::Type::Incremented>(compensations_port, 0)}},
                               {k_block, {LoopPort::create<LoopPort::Type::NotIncremented>(compensations_port, 1)}}});
        }
    }

    const bool with_postops = brgemm->get_input_size() - brgemm->get_gemm_inputs_count() > 0;
    if (with_postops) {
        create_not_processed_postops_ports(brgemm_expr, loop_manager, m_block, n_block, k_block);
    }
    return res;
}
}  // namespace ov::intel_cpu::pass
