// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "snippets/lowered/pass/runtime_optimizer.hpp"
#include "snippets/runtime_configurator.hpp"

namespace ov {
namespace intel_cpu {

/**
 * @class BrgemmExternalRepackingAdjuster
 * @brief A runtime optimizer that creates the memory descs for BRGEMM inputs which require external repacking.
 * The generated memory descs are stored in the CPU runtime config.
 */
class BrgemmExternalRepackingAdjuster : public ov::snippets::lowered::pass::RuntimeOptimizer {
public:
    BrgemmExternalRepackingAdjuster() = default;
    BrgemmExternalRepackingAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir, const CPURuntimeConfigurator* configurator);

    bool run(const snippets::lowered::LinearIR& linear_ir) override;
    bool applicable() const override { return !m_param_idces_with_external_repacking.empty(); }

private:
    std::set<size_t> m_param_idces_with_external_repacking;
};

}   // namespace intel_cpu
}   // namespace ov
