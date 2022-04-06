// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/configuration/options/memory_type.hpp"
#include "vpu/utils/ddr_type.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

#include <vpu/myriad_config.hpp>

#include <unordered_map>

namespace vpu {

namespace {

const std::unordered_map<std::string, MovidiusDdrType>& string2type() {
    static const std::unordered_map<std::string, MovidiusDdrType> converters = {
        {InferenceEngine::MYRIAD_DDR_AUTO,         MovidiusDdrType::AUTO },
        {InferenceEngine::MYRIAD_DDR_MICRON_2GB,   MovidiusDdrType::MICRON_2GB },
        {InferenceEngine::MYRIAD_DDR_SAMSUNG_2GB,  MovidiusDdrType::SAMSUNG_2GB },
        {InferenceEngine::MYRIAD_DDR_HYNIX_2GB,    MovidiusDdrType::HYNIX_2GB },
        {InferenceEngine::MYRIAD_DDR_MICRON_1GB,   MovidiusDdrType::MICRON_1GB },
    };
    return converters;
}

}  // namespace

void MemoryTypeOption::validate(const std::string& value) {
    const auto& converters = string2type();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void MemoryTypeOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string MemoryTypeOption::key() {
    return InferenceEngine::MYRIAD_DDR_TYPE;
}

details::Access MemoryTypeOption::access() {
    return details::Access::Public;
}

details::Category MemoryTypeOption::category() {
    return details::Category::RunTime;
}

std::string MemoryTypeOption::defaultValue() {
    return InferenceEngine::MYRIAD_DDR_AUTO;
}

MemoryTypeOption::value_type MemoryTypeOption::parse(const std::string& value) {
    const auto& converters = string2type();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
