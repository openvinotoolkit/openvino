// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/configuration/options/platform.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

#include <vpu/myriad_config.hpp>
#include <vpu/myriad_plugin_config.hpp>

#include <unordered_map>

namespace vpu {

namespace {

const std::unordered_map<std::string, ncDevicePlatform_t>& string2platform() {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_map<std::string, ncDevicePlatform_t> converters = {
        {VPU_MYRIAD_CONFIG_VALUE(2450), ncDevicePlatform_t::NC_MYRIAD_2},
        {VPU_MYRIAD_CONFIG_VALUE(2480), ncDevicePlatform_t::NC_MYRIAD_X},
        {std::string(),                 ncDevicePlatform_t::NC_ANY_PLATFORM},
    };
IE_SUPPRESS_DEPRECATED_END
    return converters;
}

}  // namespace

void PlatformOption::validate(const std::string& value) {
    const auto& converters = string2platform();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void PlatformOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string PlatformOption::key() {
IE_SUPPRESS_DEPRECATED_START
    return VPU_MYRIAD_CONFIG_KEY(PLATFORM);
IE_SUPPRESS_DEPRECATED_END
}

details::Access PlatformOption::access() {
    return details::Access::Public;
}

details::Category PlatformOption::category() {
    return details::Category::RunTime;
}

std::string PlatformOption::defaultValue() {
    return std::string();
}

PlatformOption::value_type PlatformOption::parse(const std::string& value) {
    const auto& converters = string2platform();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
