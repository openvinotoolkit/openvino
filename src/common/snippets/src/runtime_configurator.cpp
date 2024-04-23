// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/runtime_configurator.hpp"

#include "snippets/utils.hpp"
#include "snippets/lowered/pass/update_loop_info.hpp"

namespace ov {
namespace snippets {

RuntimeConfigurator::RuntimeConfigurator(std::shared_ptr<RuntimeConfig> c) : m_config(std::move(c)) {
    OPENVINO_ASSERT(m_config, "Runtime config is nullptr!");

    // Init LinearIR StateUpdater: some passes to update LoopInfo, BufferInfo etc
    m_state_updater = lowered::pass::PassPipeline();
    m_state_updater.register_pass<lowered::pass::UpdateLoopInfo>();
}

const std::shared_ptr<RuntimeConfig>& RuntimeConfigurator::get_updated_config(const std::shared_ptr<lowered::LinearIR>& linear_ir) {
    if (is_update_needed(linear_ir))
        update(linear_ir);
    return m_config;
}

void RuntimeConfigurator::update_linear_ir_state(const std::shared_ptr<lowered::LinearIR>& linear_ir) const {
    m_state_updater.run(*linear_ir);
}

} // namespace snippets
} // namespace ov
