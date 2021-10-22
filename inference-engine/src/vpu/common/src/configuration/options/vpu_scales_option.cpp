// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/configuration/options/vpu_scales_option.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void VPUScalesOption::validate(const std::string& value) {}

void VPUScalesOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string VPUScalesOption::key() {
    return InferenceEngine::MYRIAD_SCALES_PATTERN;
}

details::Access VPUScalesOption::access() {
    return details::Access::Private;
}

details::Category VPUScalesOption::category() {
    return details::Category::CompileTime;
}

std::string VPUScalesOption::defaultValue() {
    return std::string();
}

VPUScalesOption::value_type VPUScalesOption::parse(const std::string& value) {
    std::vector<std::string> parsedStrings;
    value_type vpuScalesMap;
    int paramBeginIdx = 0;
    int paramEndIdx = 0;
    for (; paramEndIdx < value.size(); ++paramEndIdx) {
        if (value[paramEndIdx] == ';') {
            parsedStrings.push_back(value.substr(paramBeginIdx, paramEndIdx - paramBeginIdx));
            paramBeginIdx = paramEndIdx + 1;
        }
    }
    if (paramBeginIdx != paramEndIdx) {
        parsedStrings.push_back(value.substr(paramBeginIdx));
    }
    for (auto& paramStr : parsedStrings) {
        paramStr.erase(
            std::remove_if(paramStr.begin(), paramStr.end(), ::isspace),
            paramStr.end());
    }

    parsedStrings.erase(
        std::remove_if(parsedStrings.begin(), parsedStrings.end(),
                       [](std::string str) { return str.empty(); }),
        parsedStrings.end());
    std::map<std::string, double> vpuScaleMap;
    for (const auto vpuScale : parsedStrings) {
        const auto delimeterPos = vpuScale.find(':');
        std::pair<std::string, double> pair;
        pair.first = std::string(vpuScale.substr(0, delimeterPos));
        pair.second = std::stod(vpuScale.substr(delimeterPos + 1));
        vpuScaleMap.insert(pair);
    }

    return vpuScaleMap;
}

}  // namespace vpu
