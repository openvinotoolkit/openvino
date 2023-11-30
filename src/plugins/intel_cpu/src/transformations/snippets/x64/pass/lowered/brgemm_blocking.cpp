// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_blocking.hpp"

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"


namespace ov {
namespace intel_cpu {
namespace pass {
using LoopManager = snippets::lowered::LinearIR::LoopManager;
using LoopInfoPtr = LoopManager::LoopInfoPtr;
using LoopPort = LoopManager::LoopPort;

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

bool BrgemmBlocking::run(snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BrgemmBlocking")
    if (linear_ir.empty())
        return false;


    const auto& loop_manager = linear_ir.get_loop_manager();
    const size_t dim_idx = 1;

    auto blocking_loop_exists = [&](const ov::snippets::lowered::ExpressionPtr& expr,
                                    const std::shared_ptr<ov::intel_cpu::BrgemmCPU>& brgemm) {
        const auto& loop_ids = expr->get_loop_ids();
        for (const auto& id : loop_ids) {
            const auto loop = loop_manager->get_loop_info(id);
            if (loop->dim_idx == dim_idx) {
                OPENVINO_ASSERT(brgemm->get_input_count(0) == loop->increment,
                                "Brgemm ", brgemm, " has input count (", brgemm->get_input_count(0),
                                ") which doesn't match the increment(", loop->increment, ") of loop by M");
                return true;
            }
        }
        return false;
    };

    bool modified = false;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& expr = *expr_it;
        const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
        if (!brgemm || blocking_loop_exists(expr, brgemm))
            continue;

        const auto& input_shape_0 = expr->get_input_port_descriptor(0)->get_shape();
        const auto& input_layout_0 = expr->get_input_port_descriptor(0)->get_layout();
        const auto& dim = *(input_layout_0.rbegin() + dim_idx);
        const auto& m = input_shape_0[dim];

        const auto block_size = brgemm->get_m_block_size();
        brgemm->set_input_count(block_size);

        const auto work_amount = m;
        const auto increment = block_size;

        auto loop_begin_it = expr_it, loop_end_it = std::next(expr_it);
        std::vector<LoopPort> entries{LoopPort(expr->get_input_port(0), true), LoopPort(expr->get_input_port(1), false)};
        // Scratchpad for AMX scenario is needed only as temporary buffer for each M block - it means that the Buffer should be in this loop.
        // Other scratchpads (that after BrgemmCopyB) should be the loop outside.
        if (brgemm->is_with_compensations()) {
            entries.emplace_back(expr->get_input_port(2), false);
        } else if (brgemm->is_amx()) {
            move_new_memory_buffer(linear_ir, expr_it);
            loop_begin_it = std::prev(expr_it);
        }
        std::vector<LoopPort> exits{LoopPort(expr->get_output_port(0), true)};
        loop_manager->mark_loop(loop_begin_it, loop_end_it, work_amount, increment, dim_idx, entries, exits);
    }

    return modified;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov