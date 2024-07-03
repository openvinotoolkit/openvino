// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/runtime_configurator.hpp"

#include "snippets/lowered/port_descriptor.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"

namespace ov {
namespace intel_cpu {

class CPURuntimeConfig : public ov::snippets::RuntimeConfig {
public:
    OPENVINO_RTTI("CPURuntimeConfig", "0", ov::snippets::RuntimeConfig)
    CPURuntimeConfig() = default;

    std::vector<jit_snippets_call_args::loop_args_t> loop_args = {};
};

class CPURuntimeConfigurator : public ov::snippets::RuntimeConfigurator {
public:
    CPURuntimeConfigurator();

protected:
    /**
     * @brief Update RuntimeConfig based on LinearIR
     * @param linear_ir LinearIR
     */
    void update(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) override;
    /**
     * @brief Allocate and intialize fields in RuntimeConfig and RuntimeConfigurator
     * @param linear_ir LinearIR
     */
    void initialization(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) override;
    /**
     * @brief Initializes tensor rank of config
     * @param linear_ir LinearIR
     */
    void init_tensor_rank(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) const override;
    /**
     * @brief Calculate Loop parameters of Loop emitters and update these values in CPURuntimeConfig
     * @param linear_ir LinearIR
     */
    void update_loop_args(const ov::snippets::lowered::LoopManagerPtr& loop_manager) const;
    /**
     * @brief Update subtensors of Brgemms
     */
    void update_brgemms(const ov::snippets::lowered::LoopManagerPtr& loop_manager) const;

    const size_t rank6D = 6;
    std::vector<ov::snippets::lowered::ExpressionPtr> m_dynamic_brgemms = {};
};

}   // namespace intel_cpu
}   // namespace ov
