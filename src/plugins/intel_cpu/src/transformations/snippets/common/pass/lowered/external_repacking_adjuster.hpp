// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>

#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/runtime_optimizer.hpp"

namespace ov::intel_cpu::pass {

class ExternalRepackingAdjusterBase : public ov::snippets::lowered::pass::RuntimeOptimizer {
public:
    bool run(const snippets::lowered::LinearIR& linear_ir) override;
    bool applicable() const override {
        return !m_repacked_inputs.empty();
    }

protected:
    ExternalRepackingAdjusterBase() = default;
    ExternalRepackingAdjusterBase(const CPURuntimeConfigurator* configurator, std::string itt_name);

    void register_repacked_input(size_t idx, bool needs_runtime_repacking);

private:
    virtual size_t update_runtime_repacking_data_size(const snippets::lowered::LinearIR& linear_ir,
                                                      const CPURuntimeConfig& cpu_config,
                                                      size_t idx) = 0;
    virtual void update_runtime_repacking_input(const snippets::lowered::LinearIR& linear_ir,
                                                CPURuntimeConfig& cpu_config,
                                                size_t idx,
                                                bool is_impl_parallel) = 0;
    virtual void update_compile_time_repacked_input(const snippets::lowered::LinearIR& linear_ir,
                                                    CPURuntimeConfig& cpu_config,
                                                    size_t idx) = 0;

    std::string m_itt_name;
    std::unordered_map<size_t, bool> m_repacked_inputs;
};

}  // namespace ov::intel_cpu::pass
