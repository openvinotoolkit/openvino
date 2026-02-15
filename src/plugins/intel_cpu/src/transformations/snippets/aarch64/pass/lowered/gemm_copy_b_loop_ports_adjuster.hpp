// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_map>
#include <vector>

#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "openvino/core/rtti.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/pass/runtime_optimizer.hpp"

namespace ov::intel_cpu::pass::aarch64 {

/**
 * @class GemmCopyBLoopPortsAdjuster
 * @brief A runtime optimizer that adjusts blocked loops parameters for gemm operations which require repacking.
 */
class GemmCopyBLoopPortsAdjuster : public ov::snippets::lowered::pass::RuntimeOptimizer {
public:
    OPENVINO_RTTI("GemmCopyBLoopPortsAdjuster", "", RuntimeOptimizer)
    GemmCopyBLoopPortsAdjuster() = default;
    GemmCopyBLoopPortsAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                               const CPURuntimeConfigurator* configurator);

    bool run(const snippets::lowered::LinearIR& linear_ir) override;
    bool applicable() const override {
        return !m_affected_uni2exp_map.empty();
    }

private:
    std::unordered_map<snippets::lowered::UnifiedLoopInfoPtr, std::vector<snippets::lowered::ExpandedLoopInfoPtr>>
        m_affected_uni2exp_map;
};

}  // namespace ov::intel_cpu::pass::aarch64
