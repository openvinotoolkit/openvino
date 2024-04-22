// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace snippets {

/**
 * @interface RuntimeConfig
 * @brief The config that contains information about LinearIR in runtime.
 */
class RuntimeConfig {
public:
    RuntimeConfig() = default;
    virtual ~RuntimeConfig() = default;
};

/**
 * @interface RuntimeConfigurator
 * @brief Configure runtime config based on runtime information of LinearIR
 */
class RuntimeConfigurator {
public:
    RuntimeConfigurator(std::shared_ptr<RuntimeConfig> c);
    virtual ~RuntimeConfigurator() = default;

    virtual const std::shared_ptr<RuntimeConfig>& update(const std::shared_ptr<lowered::LinearIR>& linear_ir) = 0;

protected:
    void update_linear_ir_state(const std::shared_ptr<lowered::LinearIR>& linear_ir) const;

    std::shared_ptr<RuntimeConfig> m_config = nullptr;
    lowered::pass::PassPipeline m_state_updater;
};

} // namespace snippets
} // namespace ov
