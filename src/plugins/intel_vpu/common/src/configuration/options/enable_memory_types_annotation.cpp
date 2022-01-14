// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/enable_memory_types_annotation.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void EnableMemoryTypesAnnotationOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void EnableMemoryTypesAnnotationOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string EnableMemoryTypesAnnotationOption::key() {
    return InferenceEngine::MYRIAD_ENABLE_MEMORY_TYPES_ANNOTATION;
}

details::Access EnableMemoryTypesAnnotationOption::access() {
    return details::Access::Private;
}

details::Category EnableMemoryTypesAnnotationOption::category() {
    return details::Category::CompileTime;
}

std::string EnableMemoryTypesAnnotationOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::NO;
}

EnableMemoryTypesAnnotationOption::value_type EnableMemoryTypesAnnotationOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
