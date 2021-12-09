// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/configuration/options/protocol.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

#include <vpu/myriad_config.hpp>
#include <vpu/vpu_plugin_config.hpp>

#include <unordered_map>

namespace vpu {

namespace {

const std::unordered_map<std::string, ncDeviceProtocol_t>& string2protocol() {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_map<std::string, ncDeviceProtocol_t> converters = {
        {InferenceEngine::MYRIAD_USB,   ncDeviceProtocol_t::NC_USB},
        {InferenceEngine::MYRIAD_PCIE,  ncDeviceProtocol_t::NC_PCIE},
        {std::string(),                 ncDeviceProtocol_t::NC_ANY_PROTOCOL},

        // Deprecated
        {VPU_MYRIAD_CONFIG_VALUE(USB), ncDeviceProtocol_t::NC_USB},
        {VPU_MYRIAD_CONFIG_VALUE(PCIE),  ncDeviceProtocol_t::NC_PCIE}
    };
IE_SUPPRESS_DEPRECATED_END
    return converters;
}

}  // namespace

void ProtocolOption::validate(const std::string& value) {
    const auto& converters = string2protocol();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void ProtocolOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string ProtocolOption::key() {
    return InferenceEngine::MYRIAD_PROTOCOL;
}

details::Access ProtocolOption::access() {
    return details::Access::Public;
}

details::Category ProtocolOption::category() {
    return details::Category::RunTime;
}

std::string ProtocolOption::defaultValue() {
    return std::string();
}

ProtocolOption::value_type ProtocolOption::parse(const std::string& value) {
    const auto& converters = string2protocol();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
