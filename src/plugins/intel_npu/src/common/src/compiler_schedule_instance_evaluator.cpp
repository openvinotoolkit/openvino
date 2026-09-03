// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_schedule_instance_evaluator.hpp"

namespace {}  // namespace

namespace intel_npu {

CompilerScheduleInstanceEvaluator::CompilerScheduleInstanceEvaluator(

    const ov::SoPtr<intel_npu::IEngineBackend>& backend,
    const std::shared_ptr<CompilerOptionSupportHelper>& option_support_helper)
    : ISectionInstanceEvaluator(),
      m_backend(backend),
      m_option_support_helper(option_support_helper) {
    OPENVINO_ASSERT(backend && backend->getDevice(),
                    "A device object is required to validate the compiler requirements");
}

bool CompilerScheduleInstanceEvaluator::evaluate(std::string_view runtime_requirements) const {
    if (runtime_requirements.empty()) {
        return true;
    }

    const auto device = m_backend->getDevice();
    const auto init_structs = m_backend->getInitStructs();

    if (device != nullptr && init_structs != nullptr && init_structs->getZeDrvApiVersion() >= ZE_MAKE_VERSION(1, 16)) {
        return device->validateCompatibilityDescriptor(runtime_requirements.data());
    }

    // Fallback routed through the option support helper
    return m_option_support_helper->isOptionSupported(ov::intel_npu::CompilerType::PLUGIN,
                                                      ov::compatibility_check.name(),
                                                      std::make_optional(std::string(runtime_requirements)));
}

}  // namespace intel_npu
