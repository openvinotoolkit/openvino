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
#include "transformations/snippets/tpp/op/brgemm.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {
using LinearIR = snippets::lowered::LinearIR;
using LoopPort = LinearIR::LoopManager::LoopPort;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;
using LoopInfo = LinearIR::LoopManager::LoopInfo;
using namespace ov::snippets::lowered::pass;

BrgemmBlocking::BrgemmBlocking() : RangedPass() {}

void BrgemmBlocking::move_new_memory_buffer(snippets::lowered::LinearIR& linear_ir, const snippets::lowered::LinearIR::constExprIt& brgemm_it) {
    const auto& brgemm_expr = brgemm_it->get();
    const auto wsp_expr = brgemm_expr->get_input_port_connector(2)->get_source().get_expr();
    const auto wsp_buffer = ov::as_type_ptr<ov::snippets::op::NewMemoryBuffer>(wsp_expr->get_node());
    OPENVINO_ASSERT(wsp_buffer, "Incorrect Scratchpad buffer for Brgemm AMX");
    // If scratchpad with temp memory is not explicitly before Brgemm, need to move to there.
    if (wsp_expr != *std::prev(brgemm_it)) {
        const auto wsp_it = linear_ir.find(wsp_expr);
        linear_ir.move(wsp_it, brgemm_it);
    }
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
            if (std::any_of(loop->get_entry_points().begin(), loop->get_entry_points().end(), check_port) ||
                std::any_of(loop->get_exit_points().begin(), loop->get_exit_points().end(), check_port)) {
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
                        "Detected invalid Brgemm operation: ops must be assigned to a backed when blocking is performed.");

        const auto& in_0_desc = brgemm_expr->get_input_port_descriptor(0);
        const auto& in_1_desc = brgemm_expr->get_input_port_descriptor(1);
        const auto& out_desc = brgemm_expr->get_output_port_descriptor(0);

        const auto& in_0_planar_dims = ov::snippets::utils::get_planar_vdims(in_0_desc->get_shape(), in_0_desc->get_layout());
        const auto& in_1_planar_dims = ov::snippets::utils::get_planar_vdims(in_1_desc->get_shape(), in_1_desc->get_layout());
        const auto& out_preordered_dims = ov::snippets::utils::get_preordered_vdims(out_desc->get_shape(), out_desc->get_layout());

        auto in_0_subtensor = in_0_desc->get_subtensor();
        auto in_1_subtensor = in_1_desc->get_subtensor();
        auto out_subtensor = out_desc->get_subtensor();

        auto apply_m_blocking = [&]() {
            const auto& m = *(out_preordered_dims.rbegin() + 1);
            const auto block_size_m = brgemm->get_m_block_size();
            if (block_size_m >= m) {
                *(in_0_subtensor.rbegin() + 1) = m;
                *(out_subtensor.rbegin() + 1) = m;
                return;
            }

            *(in_0_subtensor.rbegin() + 1) = block_size_m;
            *(out_subtensor.rbegin() + 1) = block_size_m;
            auto loop_begin_it = expr_it, loop_end_it = std::next(expr_it);
            std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), true),
                                          LoopPort(brgemm_expr->get_input_port(1), false)};
            if (brgemm_cpu) {
                if (brgemm_cpu->is_with_compensations()) {
                    entries.emplace_back(brgemm_expr->get_input_port(2), false);
                } else if (brgemm_cpu->is_amx()) {
                    move_new_memory_buffer(linear_ir, expr_it);
                    loop_begin_it = std::prev(expr_it);
                }
            }
            std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
            loop_manager->mark_loop(loop_begin_it, loop_end_it, m, block_size_m, 1, entries, exits);
        };

        auto apply_n_blocking = [&]() {
            const auto& n = *out_preordered_dims.rbegin();
            const auto block_size_n = brgemm->get_n_block_size();
            if (block_size_n >= n) {
                *in_1_subtensor.rbegin() = n;
                *out_subtensor.rbegin() = n;
                return;
            }

            *in_1_subtensor.rbegin() = block_size_n;
            *out_subtensor.rbegin() = block_size_n;
            auto loop_begin_it = expr_it, loop_end_it = std::next(expr_it);
            std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), false),
                                          LoopPort(brgemm_expr->get_input_port(1), true)};
            if (brgemm_cpu) {
                if (brgemm_cpu->is_with_compensations()) {
                    entries.emplace_back(brgemm_expr->get_input_port(2), true);
                } else if (brgemm_cpu->is_amx()) {
                    move_new_memory_buffer(linear_ir, expr_it);
                    loop_begin_it = std::prev(expr_it);
                }
            }
            std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
            loop_manager->mark_loop(loop_begin_it, loop_end_it, n, block_size_n, 0, entries, exits);
        };

        auto apply_k_blocking = [&]() {
            const auto& k = *in_0_planar_dims.rbegin();
            OPENVINO_ASSERT(k == *(in_1_planar_dims.rbegin() + 1), "Brgemm input descriptors have different K dimension value.");
            const auto block_size_k = brgemm->get_k_block_size();
            if (block_size_k >= k) {
                *in_0_subtensor.rbegin() = k;
                *(in_1_subtensor.rbegin() + 1) = k;
                brgemm->set_beta(0.f);
                return;
            }

            *in_0_subtensor.rbegin() = block_size_k;
            *(in_1_subtensor.rbegin() + 1) = block_size_k;
            auto loop_begin_it = expr_it, loop_end_it = std::next(expr_it);
            std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), true, 0),
                                          LoopPort(brgemm_expr->get_input_port(1), true, 1)};
            if (brgemm_cpu) {
                if (brgemm_cpu->is_with_compensations()) {
                    entries.emplace_back(brgemm_expr->get_input_port(2), false, 1);
                } else if (brgemm_cpu->is_amx()) {
                    move_new_memory_buffer(linear_ir, expr_it);
                    loop_begin_it = std::prev(expr_it);
                }
            }
            std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), false)};
            const auto id = loop_manager->mark_loop(loop_begin_it, loop_end_it, k, block_size_k, entries, exits);
            const auto loop_info = loop_manager->get_loop_info(id);
            loop_info->register_handler<LoopInfo::SpecificIterationHandlers::HandlerType::FIRST_ITER, SetBrgemmBeta>(0.f);
        };

        apply_k_blocking();
        apply_n_blocking();
        apply_m_blocking();

        brgemm_expr->get_input_port_descriptor(0)->set_subtensor(in_0_subtensor);
        brgemm_expr->get_input_port_descriptor(1)->set_subtensor(in_1_subtensor);
        brgemm_expr->get_output_port_descriptor(0)->set_subtensor(out_subtensor);
        modified = true;
    }

    return modified;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov