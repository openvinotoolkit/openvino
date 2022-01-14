// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/configuration/options/hw_black_list.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/utils/error.hpp"

#include <vpu/utils/string.hpp>

namespace vpu {

void HwBlackListOption::validate(const std::string& value) {
    try {
        splitStringList<HwBlackListOption::value_type>(value, ',');
    } catch(const std::exception& e) {
        VPU_THROW_FORMAT(R"(unexpected {} option value "{}")", key(), value);
    }
}

void HwBlackListOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string HwBlackListOption::key() {
    return InferenceEngine::MYRIAD_HW_BLACK_LIST;
}

details::Access HwBlackListOption::access() {
    return details::Access::Private;
}

details::Category HwBlackListOption::category() {
    return details::Category::CompileTime;
}

std::string HwBlackListOption::defaultValue() {
    return std::string();
}

HwBlackListOption::value_type HwBlackListOption::parse(const std::string& value) {
    HwBlackListOption::value_type stringList;
    try {
        stringList = splitStringList<HwBlackListOption::value_type>(value, ',');
    } catch(const std::exception& e) {
        VPU_THROW_FORMAT(R"(unexpected {} option value "{}")", key(), value);
    }
    return stringList;
}

}  // namespace vpu
