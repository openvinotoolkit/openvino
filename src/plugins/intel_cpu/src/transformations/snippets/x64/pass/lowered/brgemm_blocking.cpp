// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_blocking.hpp"

#include "cpu_iter_handlers.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/tpp/x64/op/brgemm.hpp"


namespace ov {
namespace intel_cpu {
namespace pass {
using LinearIR = snippets::lowered::LinearIR;
using LoopPort = snippets::lowered::LoopPort;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;
using namespace ov::snippets::lowered;

BrgemmBlocking::BrgemmBlocking() : RangedPass() {}

LinearIR::constExprIt BrgemmBlocking::move_new_memory_buffer(LinearIR& linear_ir, const LinearIR::constExprIt& brgemm_it) {
    const auto& brgemm_expr = brgemm_it->get();
    const auto wsp_expr = brgemm_expr->get_input_port_connector(2)->get_source().get_expr();
    const auto wsp_buffer = ov::as_type_ptr<ov::snippets::op::NewMemoryBuffer>(wsp_expr->get_node());
    OPENVINO_ASSERT(wsp_buffer, "Incorrect Scratchpad buffer for Brgemm AMX");
    // If scratchpad with temp memory is not explicitly before Brgemm, need to move to there.
    if (wsp_expr != *std::prev(brgemm_it)) {
        const auto wsp_it = linear_ir.find(wsp_expr);
        linear_ir.move(wsp_it, brgemm_it);
    }
    return std::prev(brgemm_it);
}

LinearIR::constExprIt BrgemmBlocking::get_loop_begin_pos(LinearIR& linear_ir, const LinearIR::constExprIt& brgemm_it) {
    auto loop_begin_it = brgemm_it;
    const auto& brgemm_expr = *brgemm_it;
    const auto node = brgemm_expr->get_node();
    const auto brgemm = ov::as_type_ptr<snippets::op::Brgemm>(node);
    const auto brgemm_cpu = ov::as_type_ptr<intel_cpu::BrgemmCPU>(node);
    OPENVINO_ASSERT(brgemm, "get_loop_begin_pos must be called only for Brgemm expression");
    if (brgemm_cpu && brgemm_cpu->is_amx()) {
        loop_begin_it = move_new_memory_buffer(linear_ir, brgemm_it);
    }
    if (brgemm_cpu && brgemm_cpu->is_with_data_repacking()) {
        const auto& copy_b = brgemm_cpu->get_brgemm_copy();
        const auto& copy_b_expr = linear_ir.get_expr_by_node(copy_b);
        loop_begin_it = linear_ir.find(copy_b_expr);
    }
    return loop_begin_it;
}

bool BrgemmBlocking::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BrgemmBlocking")
    const auto& loop_manager = linear_ir.get_loop_manager();
    auto blocking_loop_exists = [&](const ExpressionPtr& brgemm_expr, const std::shared_ptr<snippets::op::Brgemm>& brgemm) {
        auto check_port = [&](const LoopPort& p) {
            return p.expr_port->get_expr() == brgemm_expr && ov::snippets::utils::one_of(p.dim_idx, 0ul, 1ul);
        };

        const auto& loop_ids = brgemm_expr->get_loop_ids();
        for (const auto& id : loop_ids) {
            const auto loop = loop_manager->get_loop_info(id);
            if (std::any_of(loop->get_input_ports().begin(), loop->get_input_ports().end(), check_port) ||
                std::any_of(loop->get_output_ports().begin(), loop->get_output_ports().end(), check_port)) {
                return true;
            }
        }
        return false;
    };

    bool modified = false;
    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& brgemm_expr = *expr_it;
        const auto& node = brgemm_expr->get_node();
        const auto brgemm = ov::as_type_ptr<snippets::op::Brgemm>(node);
        const auto brgemm_cpu = ov::as_type_ptr<intel_cpu::BrgemmCPU>(node);
        if (!brgemm || blocking_loop_exists(brgemm_expr, brgemm))
            continue;
        OPENVINO_ASSERT(ov::is_type<intel_cpu::BrgemmCPU>(node) || ov::is_type<intel_cpu::tpp::op::BrgemmTPP>(node),
                        "Detected invalid Brgemm operation: ops must be assigned to a backend when blocking is performed.");

        const auto& in_0_desc = brgemm_expr->get_input_port_descriptor(0);
        const auto& in_1_desc = brgemm_expr->get_input_port_descriptor(1);
        const auto& out_desc = brgemm_expr->get_output_port_descriptor(0);

        const auto& in_0_planar_dims = ov::snippets::utils::get_planar_vdims(in_0_desc->get_shape(), in_0_desc->get_layout());
        const auto& in_1_planar_dims = ov::snippets::utils::get_planar_vdims(in_1_desc->get_shape(), in_1_desc->get_layout());
        const auto& out_preordered_dims = ov::snippets::utils::get_preordered_vdims(out_desc->get_shape(), out_desc->get_layout());

        auto in_0_subtensor = in_0_desc->get_subtensor();
        auto in_1_subtensor = in_1_desc->get_subtensor();
        auto out_subtensor = out_desc->get_subtensor();

        const auto& m = *++out_preordered_dims.rbegin();
        const auto& n = *out_preordered_dims.rbegin();
        const auto& k = *in_0_planar_dims.rbegin();
        OPENVINO_ASSERT(k == *++in_1_planar_dims.rbegin(), "Brgemm input descriptors have different K dimension value.");

        const auto block_size_m = snippets::utils::is_dynamic_value(m) ? brgemm->get_m_block_size() : std::min(brgemm->get_m_block_size(), m);
        const auto block_size_n = snippets::utils::is_dynamic_value(n) ? brgemm->get_n_block_size() : std::min(brgemm->get_n_block_size(), n);
        const auto block_size_k = snippets::utils::is_dynamic_value(k) ? brgemm->get_k_block_size() : std::min(brgemm->get_k_block_size(), k);

        *++in_0_subtensor.rbegin() = block_size_m;
        *++out_subtensor.rbegin() = block_size_m;
        *in_1_subtensor.rbegin() = block_size_n;
        *out_subtensor.rbegin() = block_size_n;
        *in_0_subtensor.rbegin() = block_size_k;
        *++in_1_subtensor.rbegin() = block_size_k;

        brgemm_expr->get_input_port_descriptor(0)->set_subtensor(in_0_subtensor);
        brgemm_expr->get_input_port_descriptor(1)->set_subtensor(in_1_subtensor);
        brgemm_expr->get_output_port_descriptor(0)->set_subtensor(out_subtensor);

        ov::snippets::lowered::ExpressionPtr copy_b_expr = nullptr;
        if (brgemm_cpu && brgemm_cpu->is_with_data_repacking()) {
            const auto copy_b = brgemm_cpu->get_brgemm_copy();
            copy_b_expr = linear_ir.get_expr_by_node(copy_b);

            auto data_repacking_subtensor = copy_b_expr->get_input_port_descriptor(0)->get_subtensor();
            *data_repacking_subtensor.rbegin() = block_size_n;
            *++data_repacking_subtensor.rbegin() = block_size_k;

            copy_b_expr->get_input_port_descriptor(0)->set_subtensor(data_repacking_subtensor);
            copy_b_expr->get_output_port_descriptor(0)->set_subtensor(data_repacking_subtensor);
            if (copy_b->is_with_compensations()) {
                auto compensations_subtensor = copy_b_expr->get_output_port_descriptor(1)->get_subtensor();
                // Compensations are computed by N dimension
                *compensations_subtensor.rbegin() = block_size_n;
                *++compensations_subtensor.rbegin() = 1;
                copy_b_expr->get_output_port_descriptor(1)->set_subtensor(compensations_subtensor);
            }
        }

        auto mark_m_blocking = [&]() {
            const auto loop_begin_it = get_loop_begin_pos(linear_ir, expr_it);
            const auto loop_end_it = std::next(expr_it);

            const std::vector<LoopPort> entries{
                LoopPort(brgemm_expr->get_input_port(0), true),
                LoopPort(brgemm_cpu && brgemm_cpu->is_with_data_repacking() ? copy_b_expr->get_input_port(0) : brgemm_expr->get_input_port(1), false)};
            const std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
            loop_manager->mark_loop(loop_begin_it, loop_end_it, m, block_size_m, 1, entries, exits);
        };

        auto mark_n_blocking = [&]() {
            const auto loop_begin_it = get_loop_begin_pos(linear_ir, expr_it);
            const auto loop_end_it = std::next(expr_it);

            const std::vector<LoopPort> entries{
                LoopPort(brgemm_expr->get_input_port(0), false),
                LoopPort(brgemm_cpu && brgemm_cpu->is_with_data_repacking() ? copy_b_expr->get_input_port(0) : brgemm_expr->get_input_port(1), true)};
            const std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
            loop_manager->mark_loop(loop_begin_it, loop_end_it, n, block_size_n, 0, entries, exits);
        };

        auto mark_k_blocking = [&]() {
            const auto loop_begin_it = get_loop_begin_pos(linear_ir, expr_it);
            const auto loop_end_it = std::next(expr_it);

            const std::vector<LoopPort> entries{
                LoopPort(brgemm_expr->get_input_port(0), true, 0),
                LoopPort(brgemm_cpu && brgemm_cpu->is_with_data_repacking() ? copy_b_expr->get_input_port(0) : brgemm_expr->get_input_port(1), true, 1)};
            const std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), false)};
            const auto id = loop_manager->mark_loop(loop_begin_it, loop_end_it, k, block_size_k, entries, exits);
            const auto& loop_info = loop_manager->get_loop_info<ov::snippets::lowered::UnifiedLoopInfo>(id);
            loop_info->register_pass_to_handler<ov::snippets::lowered::SpecificLoopIterType::FIRST_ITER, SetBrgemmBeta>(0.f);
        };

        if (block_size_k != k) {
            mark_k_blocking();
        } else {
            brgemm->set_beta(0.f);
        }
        if (block_size_n != n)
            mark_n_blocking();
        if (block_size_m != m)
            mark_m_blocking();
        modified = true;
    }

    return modified;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov