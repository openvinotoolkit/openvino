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

bool BrgemmBlocking::run(snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BrgemmBlocking")
    if (linear_ir.empty())
        return false;

    bool modified = false;
    const auto& loop_manager = linear_ir.get_loop_manager();

    auto brgemm_m = ov::pass::pattern::wrap_type<ov::intel_cpu::BrgemmCPU>();
    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(brgemm_m, "BrgemmLowered");
    // Ticket: 113745
    // TODO: make the block size configurable
    const auto m_block_size = 32;
    const auto m_dim_idx = 1;

    auto blocking_loop_exists = [&](const ov::snippets::lowered::ExpressionPtr expr,
                                    const std::shared_ptr<ov::intel_cpu::BrgemmCPU> brgemm) {
        const auto& loop_ids = expr->get_loop_ids();
        for (const auto& id : loop_ids) {
            const auto loop = loop_manager->get_loop_info(id);
            if (loop->dim_idx == m_dim_idx) {
                OPENVINO_ASSERT(brgemm->get_input_count(0) == loop->increment,
                                "Brgemm ", brgemm, " has input count (", brgemm->get_input_count(0),
                                ") which doesn't match the increment(", loop->increment, ") of loop by M");
                return true;
            }
        }
        return false;
    };

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& expr = *expr_it;
        const auto node = expr->get_node();
        if (!matcher->match(node))
            continue;

        const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(node);
        if (blocking_loop_exists(expr, brgemm))
            continue;

        const auto& input_shape_0 = expr->get_input_port_descriptor(0)->get_shape();
        const auto& input_layout_0 = expr->get_input_port_descriptor(0)->get_layout();
        const auto& dim = *(input_layout_0.rbegin() + m_dim_idx);
        const auto& m = input_shape_0[dim];

        brgemm->set_input_count(m_block_size);

        const auto work_amount = m;
        const auto increment = m_block_size;

        std::vector<LoopPort> entries{LoopPort(expr->get_input_port(0), true), LoopPort(expr->get_input_port(1), false)};
        if (brgemm->is_with_scratchpad())
            entries.emplace_back(expr->get_input_port(2), false);
        std::vector<LoopPort> exits{LoopPort(expr->get_output_port(0), true)};
        loop_manager->mark_loop(expr_it, std::next(expr_it), work_amount, increment, m_dim_idx, entries, exits);
    }

    return modified;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov