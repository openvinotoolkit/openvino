// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <vector>

#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "openvino/core/rtti.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/runtime_optimizer.hpp"

namespace ov::intel_cpu::pass::aarch64 {

/**
 * @class GemmExternalRepackingAdjuster
 * @brief A runtime optimizer that adjusts input offsets for aarch64 GEMM inputs pre-packed at compile time.
 */
class GemmExternalRepackingAdjuster : public ov::snippets::lowered::pass::RuntimeOptimizer {
public:
    OPENVINO_RTTI("GemmExternalRepackingAdjuster", "", RuntimeOptimizer)
    GemmExternalRepackingAdjuster() = default;
    GemmExternalRepackingAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                  const CPURuntimeConfigurator* configurator);

    bool run(const snippets::lowered::LinearIR& linear_ir) override;
    bool applicable() const override {
        return !m_repacked_inputs.empty();
    }

private:
    std::vector<size_t> m_repacked_inputs;
};

}  // namespace ov::intel_cpu::pass::aarch64
