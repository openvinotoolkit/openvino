// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_blocking.hpp"

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils.hpp"
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
    const auto wsp_buffer = ov::as_type_ptr<ov::snippets::op::NewMemoryBuffer>(wsp_expr->get_node());
    OPENVINO_ASSERT(wsp_buffer, "Incorrect Scratchpad buffer for Brgemm AMX");
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
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& brgemm_expr = *expr_it;
        const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(brgemm_expr->get_node());
        if (!brgemm || blocking_loop_exists(brgemm_expr, brgemm))
            continue;

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
            } else {
                *(in_0_subtensor.rbegin() + 1) = block_size_m;
                *(out_subtensor.rbegin() + 1) = block_size_m;

                auto loop_begin_it = expr_it, loop_end_it = std::next(expr_it);
                std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), true),
                                              LoopPort(brgemm_expr->get_input_port(1), false)};
                if (brgemm->is_with_compensations()) {
                    entries.emplace_back(brgemm_expr->get_input_port(2), false);
                } else if (brgemm->is_amx()) {
                    move_new_memory_buffer(linear_ir, expr_it);
                    loop_begin_it = std::prev(expr_it);
                }
                std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
                loop_manager->mark_loop(loop_begin_it, loop_end_it, m, block_size_m, 1, entries, exits);
            }
        };

        auto apply_n_blocking = [&]() {
            const auto& n = *out_preordered_dims.rbegin();
            const auto block_size_n = brgemm->get_n_block_size();
            if (block_size_n >= n) {
                *in_1_subtensor.rbegin() = n;
                *out_subtensor.rbegin() = n;
            } else {
                *in_1_subtensor.rbegin() = block_size_n;
                *out_subtensor.rbegin() = block_size_n;

                auto loop_begin_it = expr_it, loop_end_it = std::next(expr_it);
                std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), false),
                                              LoopPort(brgemm_expr->get_input_port(1), true)};
                if (brgemm->is_with_compensations()) {
                    entries.emplace_back(brgemm_expr->get_input_port(2), true);
                } else if (brgemm->is_amx()) {
                    move_new_memory_buffer(linear_ir, expr_it);
                    loop_begin_it = std::prev(expr_it);
                }
                std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
                loop_manager->mark_loop(loop_begin_it, loop_end_it, n, block_size_n, 0, entries, exits);
            }
        };

        auto apply_k_blocking = [&]() {
            const auto& k = *in_0_planar_dims.rbegin();
            OPENVINO_ASSERT(k == *(in_1_planar_dims.rbegin() + 1), "Brgemm input descriptors have different K dimension value.");
            const auto block_size_k = brgemm->get_k_block_size();
            if (block_size_k >= k) {
                *in_0_subtensor.rbegin() = k;
                *(in_1_subtensor.rbegin() + 1) = k;
            } else {
                *in_0_subtensor.rbegin() = block_size_k;
                *(in_1_subtensor.rbegin() + 1) = block_size_k;

                auto loop_begin_it = expr_it, loop_end_it = std::next(expr_it);
                std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), true, 0),
                                              LoopPort(brgemm_expr->get_input_port(1), true, 1)};
                if (brgemm->is_with_compensations()) {
                    entries.emplace_back(brgemm_expr->get_input_port(2), false, 1);
                } else if (brgemm->is_amx()) {
                    move_new_memory_buffer(linear_ir, expr_it);
                    loop_begin_it = std::prev(expr_it);
                }
                std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), false)};
                auto loop_id = loop_manager->mark_loop(loop_begin_it, loop_end_it, k, block_size_k, entries, exits);
                const auto loop_info = loop_manager->get_loop_info(loop_id);

                auto first_iter_handler = [](LinearIR& linear_ir, LinearIR::constExprIt loop_end_it) {
                    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(loop_end_it->get()->get_node());
                    OPENVINO_ASSERT(loop_end, "First loop iteraton handler must be called on LoopEnd expression");
                    const auto loop_id = loop_end->get_id();
                    const auto& loop_manager = linear_ir.get_loop_manager();
                    const auto& loop_info = loop_manager->get_loop_info(loop_id);
                    const auto work_amount = loop_info->get_work_amount();
                    const auto increment = loop_info->get_increment();
                    if (work_amount <= increment)
                        return false;

                    const auto loop_begin_it = linear_ir.find(linear_ir.get_expr_by_node(loop_end->get_loop_begin()));
                    const auto new_loop_begin_pos = snippets::lowered::pass::InsertTailLoop::insert_copy_loop(linear_ir, loop_id, loop_begin_it);
                    const auto new_loop_begin = ov::as_type_ptr<snippets::op::LoopBegin>(new_loop_begin_pos->get()->get_node());
                    OPENVINO_ASSERT(new_loop_begin, "Cloned Loop does not contain LoopBegin op at the expected place.");
                    const auto firt_iter_loop_end = new_loop_begin->get_loop_end();
                    auto first_iter_loop_info = loop_manager->get_loop_info(firt_iter_loop_end->get_id());
                    firt_iter_loop_end->set_work_amount(increment);
                    first_iter_loop_info->set_work_amount(increment);
                    firt_iter_loop_end->set_finalization_offsets(std::vector<int64_t>(loop_end->get_finalization_offsets().size(), 0));

                    const auto new_work_amount = work_amount - increment;
                    loop_info->set_work_amount(new_work_amount);
                    loop_end->set_work_amount(new_work_amount);

                    // Update original body's Brgemms with new beta parameter
                    for (auto expr_it = loop_begin_it; expr_it != loop_end_it; ++expr_it) {
                        const auto& expr_node = expr_it->get()->get_node();
                        if (const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr_node)) {
                            brgemm->set_beta(1.f);
                        }
                    }
                    return true;
                };
                loop_info->set_first_iter_handler(first_iter_handler);
            }
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