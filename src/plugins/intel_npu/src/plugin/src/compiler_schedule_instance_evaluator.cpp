// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_schedule_instance_evaluator.hpp"

namespace {

constexpr size_t LIST_DELIMITERS_SIZE = 2;
constexpr char LIST_START_DELIMITER = '[';
constexpr char LIST_END_DELIMITER = ']';

ov::CompatibilityCheck bool_to_compatibility_check(const bool input) {
    return input ? ov::CompatibilityCheck::SUPPORTED : ov::CompatibilityCheck::UNSUPPORTED;
}

}  // namespace

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

ov::CompatibilityCheck CompilerScheduleInstanceEvaluator::evaluate(std::string_view runtime_requirements) const {
    if (runtime_requirements.size() >= LIST_DELIMITERS_SIZE && runtime_requirements.front() == LIST_START_DELIMITER &&
        runtime_requirements.back() == LIST_END_DELIMITER) {
        runtime_requirements = runtime_requirements.substr(1, runtime_requirements.size() - 2);
    }
    if (runtime_requirements.empty()) {
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }

    const auto device = m_backend->getDevice();
    const auto init_structs = m_backend->getInitStructs();

    if (device != nullptr && init_structs != nullptr && init_structs->getZeDrvApiVersion() >= ZE_MAKE_VERSION(1, 16)) {
        return bool_to_compatibility_check(device->validateCompatibilityDescriptor(std::string(runtime_requirements)));
    }

    // Fallback routed through the option support helper
    return bool_to_compatibility_check(
        m_option_support_helper->isOptionSupported(ov::intel_npu::CompilerType::PLUGIN,
                                                   ov::compatibility_check.name(),
                                                   std::make_optional(std::string(runtime_requirements))));
}

}  // namespace intel_npu
