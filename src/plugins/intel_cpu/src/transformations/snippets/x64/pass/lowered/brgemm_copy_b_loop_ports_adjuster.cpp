// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_copy_b_loop_ports_adjuster.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "transformations/snippets/x64/pass/lowered/adjust_brgemm_copy_b_loop_ports.hpp"

namespace ov::intel_cpu {

BrgemmCopyBLoopPortsAdjuster::BrgemmCopyBLoopPortsAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                                           const CPURuntimeConfigurator* configurator)
    : ov::snippets::lowered::pass::RuntimeOptimizer(configurator) {
    if (!linear_ir->is_dynamic()) {
        return;
    }

    const auto& pass = std::make_shared<intel_cpu::pass::AdjustBrgemmCopyBLoopPorts>();
    pass->run(*linear_ir);
    const auto& affected_uni_loops = pass->get_affected_loops();
    const auto& loop_map = linear_ir->get_loop_manager()->get_map();
    for (const auto& p : loop_map) {
        if (const auto& exp_loop = ov::as_type_ptr<snippets::lowered::ExpandedLoopInfo>(p.second)) {
            const auto& uni_loop = exp_loop->get_unified_loop_info();
            if (affected_uni_loops.count(uni_loop)) {
                m_affected_uni2exp_map[uni_loop].push_back(exp_loop);
            }
        }
    }
}

bool BrgemmCopyBLoopPortsAdjuster::run(const snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BrgemmCopyBLoopPortsAdjuster")
    for (const auto& p : m_affected_uni2exp_map) {
        const auto& uni_loop = p.first;
        const auto& exp_loops = p.second;
        snippets::RuntimeConfigurator::LoopInfoRuntimeParamsMap initialized_info;
        if (intel_cpu::pass::AdjustBrgemmCopyBLoopPorts::update_loop_info(uni_loop)) {
            initialized_info[uni_loop] = snippets::RuntimeConfigurator::get_loop_runtime_params(uni_loop);
            for (const auto& exp_loop : exp_loops) {
                snippets::RuntimeConfigurator::update_expanded_loop_info(exp_loop, initialized_info);
            }
        }
    }
    return true;
}

}  // namespace ov::intel_cpu
