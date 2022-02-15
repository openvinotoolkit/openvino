// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/configuration/options/vpu_scales_option.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 9) && !defined(__clang__) && !defined(IE_GCC_4_8)
# define IE_GCC_4_8
#else
# include <regex>
#endif

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
    value_type vpuScaleMap;
    #ifdef IE_GCC_4_8
        VPU_THROW_UNLESS(value.empty(), "It is not possible to parse the 'scale' value from the config because you are using a gcc version less than 4.9.");
    #else
    std::vector<std::string> parsedStrings;

    auto delimiterToken = std::regex(";");
    auto regexScales =  std::sregex_token_iterator(value.begin(), value.end(), delimiterToken, -1);
    std::sregex_token_iterator end;
    for ( ; regexScales != end; ++regexScales) {
        parsedStrings.push_back(*regexScales);
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
    for (const auto& vpuScale : parsedStrings) {
        const auto delimeterPos = vpuScale.find(':');
        VPU_THROW_UNLESS(delimeterPos != std::string::npos, "Unable to parse string \"{}\"", vpuScale);
        try {
            vpuScaleMap.insert({std::string(vpuScale.substr(0, delimeterPos)),
                                std::stof(vpuScale.substr(delimeterPos + 1))});
        } catch (...) {
            VPU_THROW_EXCEPTION << "Cannot convert string to float. Wrong input";
        }
    }

    #endif
    return vpuScaleMap;
}

}  // namespace vpu
