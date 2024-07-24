// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_blocking.hpp"

#include "cpu_iter_handlers.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/lowered/pass/propagate_subtensors.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
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

LinearIR::constExprIt BrgemmBlocking::get_loop_begin_pos(LinearIR& linear_ir, const LinearIR::constExprIt& brgemm_it, bool include_repacking) {
    auto loop_begin_it = brgemm_it;
    const auto& brgemm_expr = *brgemm_it;
    const auto node = brgemm_expr->get_node();
    const auto brgemm = ov::as_type_ptr<snippets::op::Brgemm>(node);
    const auto brgemm_cpu = ov::as_type_ptr<intel_cpu::BrgemmCPU>(node);
    OPENVINO_ASSERT(brgemm, "get_loop_begin_pos must be called only for Brgemm expression");
    if (brgemm_cpu && with_amx(brgemm_cpu->get_type())) {
        loop_begin_it = move_new_memory_buffer(linear_ir, brgemm_it);
    }
    if (include_repacking && brgemm_cpu && with_repacking(brgemm_cpu->get_type())) {
        const auto& copy_b = brgemm_cpu->get_brgemm_copy();
        const auto& copy_b_expr = linear_ir.get_expr_by_node(copy_b);
        loop_begin_it = linear_ir.find(copy_b_expr);
    }
    return loop_begin_it;
}

snippets::lowered::SpecificIterationHandlers BrgemmBlocking::get_default_blocking_loop_handlers(size_t work_amount, size_t block_size) {
    SpecificIterationHandlers handlers;
    const auto tail_size = snippets::utils::is_dynamic_value(work_amount) ? snippets::utils::get_dynamic_value<size_t>() : work_amount % block_size;
    if (tail_size != 0)
        handlers.register_pass<snippets::lowered::SpecificLoopIterType::LAST_ITER, snippets::lowered::pass::UpdateSubtensors>(tail_size);
    handlers.register_pass<snippets::lowered::SpecificLoopIterType::LAST_ITER, SetEvaluateOnce>();
    return handlers;
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

        // If block_size is dynamic, it means that Brgemm will process full tensor:
        //   subtensor[i] = FULL_DIM as by default
        if (!snippets::utils::is_dynamic_value(block_size_m)) {
            brgemm_expr->get_input_port_descriptor(0)->set_subtensor_dim(1, block_size_m);
            brgemm_expr->get_output_port_descriptor(0)->set_subtensor_dim(1, block_size_m);
        }
        if (!snippets::utils::is_dynamic_value(block_size_n)) {
            brgemm_expr->get_input_port_descriptor(1)->set_subtensor_dim(0, block_size_n);
            brgemm_expr->get_output_port_descriptor(0)->set_subtensor_dim(0, block_size_n);
        }
        if (!snippets::utils::is_dynamic_value(block_size_k)) {
            brgemm_expr->get_input_port_descriptor(0)->set_subtensor_dim(0, block_size_k);
            brgemm_expr->get_input_port_descriptor(1)->set_subtensor_dim(1, block_size_k);
        }

        const bool need_brgemm_copy_b = brgemm_cpu && with_repacking(brgemm_cpu->get_type());
        ov::snippets::lowered::ExpressionPtr copy_b_expr = nullptr;
        if (need_brgemm_copy_b) {
            const auto copy_b = brgemm_cpu->get_brgemm_copy();
            copy_b_expr = linear_ir.get_expr_by_node(copy_b);

            auto data_repacking_subtensor = copy_b_expr->get_input_port_descriptor(0)->get_subtensor();
            *data_repacking_subtensor.rbegin() = block_size_n;
            *++data_repacking_subtensor.rbegin() = block_size_k;

            copy_b_expr->get_input_port_descriptor(0)->set_subtensor(data_repacking_subtensor);
            copy_b_expr->get_output_port_descriptor(0)->set_subtensor(data_repacking_subtensor);
            if (with_compensations(copy_b->get_type())) {
                auto compensations_subtensor = copy_b_expr->get_output_port_descriptor(1)->get_subtensor();
                // Compensations are computed by N dimension
                *compensations_subtensor.rbegin() = block_size_n;
                *++compensations_subtensor.rbegin() = 1;

                OPENVINO_ASSERT(brgemm_expr->get_input_count() == 3, "Brgemm must have 3 inputs in case of compensations.");
                brgemm_expr->get_input_port_descriptor(2)->set_subtensor(compensations_subtensor);
                copy_b_expr->get_output_port_descriptor(1)->set_subtensor(compensations_subtensor);
            }
        }

        auto mark_m_blocking = [&](bool include_repacking) {
            const auto loop_begin_it = get_loop_begin_pos(linear_ir, expr_it, include_repacking);
            const auto loop_end_it = std::next(expr_it);

            const auto b_input_port = include_repacking && need_brgemm_copy_b
                                          ? copy_b_expr->get_input_port(0)
                                          : brgemm_expr->get_input_port(1);

            std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), true), LoopPort(b_input_port, false)};
            if (!include_repacking && brgemm_cpu && with_compensations(brgemm_cpu->get_type()))
                entries.emplace_back(brgemm_expr->get_input_port(2), false);
            const std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};

            const auto id = loop_manager->mark_loop(loop_begin_it, loop_end_it, m, block_size_m, 1, entries, exits, false);
            loop_manager->get_loop_info<ov::snippets::lowered::UnifiedLoopInfo>(id)->set_handlers(get_default_blocking_loop_handlers(m, block_size_m));
        };

        auto mark_n_blocking = [&]() {
            const auto loop_begin_it = get_loop_begin_pos(linear_ir, expr_it);
            const auto loop_end_it = std::next(expr_it);

            const std::vector<LoopPort> entries{
                LoopPort(brgemm_expr->get_input_port(0), false),
                LoopPort(need_brgemm_copy_b ? copy_b_expr->get_input_port(0) : brgemm_expr->get_input_port(1), true)};
            const std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};

            const auto id = loop_manager->mark_loop(loop_begin_it, loop_end_it, n, block_size_n, 0, entries, exits, false);
            loop_manager->get_loop_info<ov::snippets::lowered::UnifiedLoopInfo>(id)->set_handlers(get_default_blocking_loop_handlers(n, block_size_n));
        };

        auto mark_k_blocking = [&]() {
            const auto loop_begin_it = get_loop_begin_pos(linear_ir, expr_it);
            const auto loop_end_it = std::next(expr_it);

            const std::vector<LoopPort> entries{
                LoopPort(brgemm_expr->get_input_port(0), true, 0),
                LoopPort(need_brgemm_copy_b ? copy_b_expr->get_input_port(0) : brgemm_expr->get_input_port(1), true, 1)};
            const std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), false)};

            auto handlers = get_default_blocking_loop_handlers(k, block_size_k);
            handlers.register_pass<ov::snippets::lowered::SpecificLoopIterType::FIRST_ITER, SetBrgemmBeta>(0.f);

            const auto id = loop_manager->mark_loop(loop_begin_it, loop_end_it, k, block_size_k, entries, exits, false);
            loop_manager->get_loop_info<ov::snippets::lowered::UnifiedLoopInfo>(id)->set_handlers(handlers);
        };

        const bool k_blocking = block_size_k != k;
        const bool n_blocking = block_size_n != n;
        const bool m_blocking = block_size_m != m;
        // It is not necessary to include copyB in loop by M if there are no blocking by KN
        const bool include_repacking_in_loop = k_blocking || n_blocking;

        if (k_blocking) {
            mark_k_blocking();
        } else {
            brgemm->set_beta(0.f);
        }
        if (n_blocking)
            mark_n_blocking();
        if (m_blocking)
            mark_m_blocking(include_repacking_in_loop);
        modified = true;
    }

    return modified;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov