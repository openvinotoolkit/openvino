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
    // TODO: make the block size configurable
    const auto m_block_size = 32;

    for (const auto& expr : linear_ir) {
        const auto node = expr->get_node();
        if (!matcher->match(node) || !expr->get_loop_ids().empty())
            continue;

        const auto m_dim_idx = 1;
        const auto& input_shape_0 = expr->get_input_port_descriptor(0)->get_shape();
        const auto& input_layout_0 = expr->get_input_port_descriptor(0)->get_layout();
        const auto& dim = *(input_layout_0.rbegin() + m_dim_idx);
        const auto& m = input_shape_0[dim];

        ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(node)->set_input_count(m_block_size);

        const auto work_amount = m;
        const auto increment = m_block_size;

        std::vector<LoopPort> entries{LoopPort(expr->get_input_port(0), true), LoopPort(expr->get_input_port(1), false)};
        std::vector<LoopPort> exits{LoopPort(expr->get_output_port(0), true)};
        loop_manager->mark_loop(expr, work_amount, increment, m_dim_idx, entries, exits);
    }

    return modified;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov