// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_blocking.hpp"

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/insert_tail_loop.hpp"
#include "snippets/snippets_isa.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"


namespace ov {
namespace intel_cpu {
namespace pass {
using LinearIR = snippets::lowered::LinearIR;
using LoopPort = LinearIR::LoopManager::LoopPort;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

BrgemmBlocking::BrgemmBlocking() : Pass() {}

void BrgemmBlocking::move_new_memory_buffer(snippets::lowered::LinearIR& linear_ir, const snippets::lowered::LinearIR::constExprIt& brgemm_it) {
    const auto& brgemm_expr = brgemm_it->get();
    const auto wsp_expr = brgemm_expr->get_input_port_connector(2)->get_source().get_expr();
    const auto wsp_buffer = ov::as_type_ptr<ov::snippets::op::Buffer>(wsp_expr->get_node());
    OPENVINO_ASSERT(wsp_buffer && wsp_buffer->is_new_memory(), "Incorrect Scratchpad buffer for Brgemm AMX");
    // [115164] Should be fully supported by explicit loops of blocking by K, N
    OPENVINO_ASSERT(brgemm_expr->get_loop_ids().empty() && wsp_expr->get_loop_ids().empty(), "Incorrect blocking loop marking for Brgemm AMX");
    // If scratchpad with temp memory is not explicitly before Brgemm, need to move to there.
    if (wsp_expr != *std::prev(brgemm_it)) {
        const auto wsp_it = linear_ir.find(wsp_expr);
        linear_ir.move(wsp_it, brgemm_it);
    }
}

bool BrgemmBlocking::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BrgemmBlocking")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    auto blocking_loop_exists = [&](const ExpressionPtr& brgemm_expr, const std::shared_ptr<ov::intel_cpu::BrgemmCPU>& brgemm) {
        auto check_port = [&](const LoopPort& p) {
            return p.expr_port->get_expr() == brgemm_expr && (p.dim_idx == 0 || p.dim_idx == 1);
        };

        const auto& loop_ids = brgemm_expr->get_loop_ids();
        for (const auto& id : loop_ids) {
            const auto loop = loop_manager->get_loop_info(id);
            if (std::any_of(loop->entry_points.begin(), loop->entry_points.end(), check_port) ||
                std::any_of(loop->exit_points.begin(), loop->exit_points.end(), check_port)) {
                return true;
            }
        }
        return false;
    };

    bool modified = false;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& brgemm_expr = *expr_it;
        const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(brgemm_expr->get_node());
        if (!brgemm || blocking_loop_exists(brgemm_expr, brgemm))
            continue;

        const auto& input_0_desc = brgemm_expr->get_input_port_descriptor(0);
        const auto& input_1_desc = brgemm_expr->get_input_port_descriptor(1);
        const auto& output_desc = brgemm_expr->get_output_port_descriptor(0);

        auto input_0_subtensor = input_0_desc->get_subtensor();
        auto input_1_subtensor = input_1_desc->get_subtensor();
        auto output_subtensor = output_desc->get_subtensor();

        auto apply_m_blocking = [&]() {
            const auto& output_shape = output_desc->get_shape();
            const auto& output_layout = output_desc->get_layout();

            const auto& m_idx = *(output_layout.rbegin() + 1);
            const auto& m = output_shape[m_idx];
            const auto block_size_m = brgemm->get_m_block_size();
            if (block_size_m >= m) {
                *(input_0_subtensor.rbegin() + 1) = m;
                *(output_subtensor.rbegin() + 1) = m;
            } else {
                *(input_0_subtensor.rbegin() + 1) = block_size_m;
                *(output_subtensor.rbegin() + 1) = block_size_m;

                std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), true), LoopPort(brgemm_expr->get_input_port(1), false)};
                if (brgemm->is_with_scratchpad())
                    entries.emplace_back(brgemm_expr->get_input_port(2), false);
                std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
                loop_manager->mark_loop(expr_it, std::next(expr_it), m, block_size_m, 1, entries, exits);
            }
        };

        auto apply_n_blocking = [&]() {
            const auto& output_shape = output_desc->get_shape();
            const auto& output_layout = output_desc->get_layout();

            const auto& n_idx = *output_layout.rbegin();
            const auto& n = output_shape[n_idx];
            const auto block_size_n = brgemm->get_n_block_size();
            if (block_size_n >= n) {
                *input_1_subtensor.rbegin() = n;
                *output_subtensor.rbegin() = n;
            } else {
                *input_1_subtensor.rbegin() = block_size_n;
                *output_subtensor.rbegin() = block_size_n;

                std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), false),
                                              LoopPort(brgemm_expr->get_input_port(1), true)};
                if (brgemm->is_with_scratchpad())
                    entries.emplace_back(brgemm_expr->get_input_port(2), true);
                std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
                loop_manager->mark_loop(expr_it, std::next(expr_it), n, block_size_n, 0, entries, exits);
            }
        };

        auto apply_k_blocking = [&]() {
            const auto& input_shape_0 = input_0_desc->get_shape();
            const auto& input_layout_0 = input_0_desc->get_layout();

            const auto& k_idx = *input_layout_0.rbegin();
            const auto& k = input_shape_0[k_idx];
            const auto block_size_k = brgemm->get_k_block_size();
            if (block_size_k >= k) {
                *input_0_subtensor.rbegin() = k;
                *(input_1_subtensor.rbegin() + 1) = k;
            } else {
                *input_0_subtensor.rbegin() = block_size_k;
                *(input_1_subtensor.rbegin() + 1) = block_size_k;

                std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), true, 0),
                                              LoopPort(brgemm_expr->get_input_port(1), true, 1)};
                if (brgemm->is_with_scratchpad())
                    entries.emplace_back(brgemm_expr->get_input_port(2), false, 1);
                std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), false)};
                auto loop_id = loop_manager->mark_loop(expr_it, std::next(expr_it), k, block_size_k, entries, exits);
                const auto loop_info = loop_manager->get_loop_info(loop_id);

                auto first_iter_handler = [](LinearIR& linear_ir, LinearIR::constExprIt expr_it) {
                    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(expr_it->get()->get_node());
                    const auto loop_id = loop_end->get_id();
                    const auto& loop_manager = linear_ir.get_loop_manager();
                    const auto& loop_info = loop_manager->get_loop_info(loop_id);
                    const auto work_amount = loop_info->work_amount;
                    const auto increment = loop_info->increment;
                    if (work_amount <= increment)
                        return false;

                    auto new_loop_range = snippets::lowered::pass::InsertTailLoop::copy_loop(linear_ir, loop_id);
                    const auto new_loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(std::prev(new_loop_range.end())->get()->get_node());
                    auto new_loop_info = loop_manager->get_loop_info(new_loop_end->get_id());
                    const auto new_work_amount = work_amount - increment;
                    new_loop_end->set_work_amount(new_work_amount);
                    new_loop_info->work_amount = new_work_amount;
                    for (const auto& expr : new_loop_range) {
                        if (const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node())) {
                            brgemm->set_beta(1.f);
                        }
                    }

                    linear_ir.insert(std::next(expr_it), new_loop_range.begin(), new_loop_range.end());

                    loop_info->work_amount = increment;
                    loop_end->set_work_amount(increment);
                    loop_end->set_finalization_offsets(std::vector<int64_t>(loop_end->get_finalization_offsets().size(), 0));
                    const auto begin_it = linear_ir.find(linear_ir.get_expr_by_node(new_loop_end->get_loop_begin()));
                    const auto end_it = linear_ir.find(linear_ir.get_expr_by_node(new_loop_end));
                    snippets::lowered::pass::InsertTailLoop::propagate_updated_subtensor_through_loop(
                        linear_ir,
                        new_loop_info,
                        std::next(begin_it),
                        end_it,
                        increment);
                    return true;
                };
                loop_info->set_first_iter_handler(first_iter_handler);
            }
        };

        apply_k_blocking();
        apply_n_blocking();
        apply_m_blocking();

        brgemm_expr->get_input_port_descriptor(0)->set_subtensor(input_0_subtensor);
        brgemm_expr->get_input_port_descriptor(1)->set_subtensor(input_1_subtensor);
        brgemm_expr->get_output_port_descriptor(0)->set_subtensor(output_subtensor);
        modified = true;
    }

    return modified;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov