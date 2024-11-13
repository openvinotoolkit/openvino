// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_runtime_configurator.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/runtime_optimizer.hpp"

namespace ov {
namespace intel_cpu {

class BrgemmCopyBLoopPortsAdjuster : public ov::snippets::lowered::pass::RuntimeOptimizer {
public:
    BrgemmCopyBLoopPortsAdjuster() = default;
    BrgemmCopyBLoopPortsAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir, CPURuntimeConfigurator* configurator);

    bool run(const snippets::lowered::LinearIR& linear_ir) override;

private:
    std::unordered_map<snippets::lowered::UnifiedLoopInfoPtr,
                       std::vector<snippets::lowered::ExpandedLoopInfoPtr>> m_affected_uni2exp_map;
};

} // namespace intel_cpu
} // namespace ov