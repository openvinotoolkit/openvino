// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_schedule_instance_evaluator.hpp"

namespace {

constexpr size_t LIST_DELIMITERS_SIZE = 2;
constexpr char LIST_START_DELIMITER = '[';
constexpr char LIST_END_DELIMITER = ']';

/**
 * @brief Allows "std::make_shared" to called the protected constructor
 */
struct MakeSharedEnabler : intel_npu::CompilerScheduleInstanceEvaluator {
    MakeSharedEnabler(const ov::SoPtr<intel_npu::IEngineBackend>& backend,
                      const std::shared_ptr<intel_npu::CompilerOptionSupportHelper>& option_support_helper)
        : intel_npu::CompilerScheduleInstanceEvaluator(backend, option_support_helper) {}
};

ov::CompatibilityCheck bool_to_compatibility_check(const bool input) {
    return input ? ov::CompatibilityCheck::SUPPORTED : ov::CompatibilityCheck::UNSUPPORTED;
}

}  // namespace

namespace intel_npu {

std::shared_ptr<CompilerScheduleInstanceEvaluator> CompilerScheduleInstanceEvaluator::get_instance(
    const ov::SoPtr<intel_npu::IEngineBackend>& backend,
    const std::shared_ptr<CompilerOptionSupportHelper>& option_support_helper) {
    OPENVINO_ASSERT(backend != nullptr && option_support_helper != nullptr, "Incomplete arguments");

    static std::mutex mutex;
    static std::weak_ptr<CompilerScheduleInstanceEvaluator> weak_instance;

    std::lock_guard<std::mutex> lock(mutex);
    auto instance = weak_instance.lock();
    if (!instance) {
        instance = std::make_shared<MakeSharedEnabler>(backend, option_support_helper);
        weak_instance = instance;
    }
    return instance;
}

CompilerScheduleInstanceEvaluator::CompilerScheduleInstanceEvaluator(
    const ov::SoPtr<intel_npu::IEngineBackend>& backend,
    const std::shared_ptr<CompilerOptionSupportHelper>& option_support_helper)
    : ISectionInstanceEvaluator(),
      m_backend(backend),
      m_option_support_helper(option_support_helper) {
    OPENVINO_ASSERT(backend && backend->getDevice(),
                    "A device object is required to validate the compiler requirements");
    OPENVINO_ASSERT(option_support_helper,
                    "A compiler option support helper object is required to validate the compiler requirements");
}

ov::CompatibilityCheck CompilerScheduleInstanceEvaluator::evaluate(std::string_view runtime_requirements) const {
    if (runtime_requirements.size() >= LIST_DELIMITERS_SIZE && runtime_requirements.front() == LIST_START_DELIMITER &&
        runtime_requirements.back() == LIST_END_DELIMITER) {
        runtime_requirements = runtime_requirements.substr(1, runtime_requirements.size() - 2);
    }
    if (runtime_requirements.empty()) {
        // Older software versions do not have this compatibility string feature. In such cases, we may reveive an
        // emptry string, which we cannot evaluate.
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }

    const auto device = m_backend->getDevice();
    const auto init_structs = m_backend->getInitStructs();

    if (device != nullptr && init_structs != nullptr && init_structs->getZeDrvApiVersion() >= ZE_MAKE_VERSION(1, 16)) {
        return bool_to_compatibility_check(device->validateCompatibilityDescriptor(std::string(runtime_requirements)));
    }

    try {
        // Fallback routed through the option support helper
        return bool_to_compatibility_check(
            m_option_support_helper->isOptionSupported(ov::intel_npu::CompilerType::PLUGIN,
                                                       ov::compatibility_check.name(),
                                                       std::make_optional(std::string(runtime_requirements))));
    } catch (...) {
        // Unable to answer
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }
}

}  // namespace intel_npu
