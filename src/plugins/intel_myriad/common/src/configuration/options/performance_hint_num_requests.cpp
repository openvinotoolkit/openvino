// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/configuration/options/performance_hint_num_requests.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/utils/error.hpp"
#include <ie_plugin_config.hpp>
#include <string>

namespace vpu {

void PerformanceHintNumRequestsOption::validate(const std::string& value) {}

void PerformanceHintNumRequestsOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string PerformanceHintNumRequestsOption::key() {
    return CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS);
}

details::Access PerformanceHintNumRequestsOption::access() {
    return details::Access::Public;
}

details::Category PerformanceHintNumRequestsOption::category() {
    return details::Category::CompileTime;
}

std::string PerformanceHintNumRequestsOption::defaultValue() {
    return "-1";
}

PerformanceHintNumRequestsOption::value_type PerformanceHintNumRequestsOption::parse(const std::string& value) {
    try {
        auto returnValue = std::stoi(value);
        if (returnValue > 0 || returnValue == -1) {
            return returnValue;
        } else {
            throw std::logic_error("wrong val");
        }
    } catch (...) {
        VPU_THROW_EXCEPTION << "Wrong value of " << value << " for property key "
                            << CONFIG_VALUE(KEY_PERFORMANCE_HINT_NUM_REQUESTS)
                            << ". Expected only positive integer numbers";
    }
}

}  // namespace vpu
