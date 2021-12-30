// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/disable_mx_boot.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/private_plugin_config.hpp"
#include <vpu/utils/string.hpp>

namespace vpu {

void DisableMXBootOption::validate(const std::string& value) {
    VPU_THROW_UNLESS(value.size() < 15, R"(unexpected {} option value "{})");
}

void DisableMXBootOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string DisableMXBootOption::key() {
    return InferenceEngine::MYRIAD_DISABLE_MX_BOOT;
}

details::Access DisableMXBootOption::access() {
    return details::Access::Private;
}

details::Category DisableMXBootOption::category() {
    return details::Category::RunTime;
}

std::string DisableMXBootOption::defaultValue() {
    return std::string("NO");
}

std::string DisableMXBootOption::parse(const std::string& value) {
    return value;
}

}  // namespace vpu
