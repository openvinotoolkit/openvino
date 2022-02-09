// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/configuration/options/performance_hint.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include <ie_plugin_config.hpp>
#include <openvino/runtime/properties.hpp>
#include <sstream>

namespace vpu {

void PerformanceHintOption::validate(const std::string& value) {}

void PerformanceHintOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string PerformanceHintOption::key() {
    return ov::hint::performance_mode.name();
}

details::Access PerformanceHintOption::access() {
    return details::Access::Public;
}

details::Category PerformanceHintOption::category() {
    return details::Category::CompileTime;
}

std::string PerformanceHintOption::defaultValue() {
    return "";
}

PerformanceHintOption::value_type PerformanceHintOption::parse(const std::string& value) {
    std::string latencyValue;
    std::string throughputValue;
    std::stringstream tmp;
    tmp << ov::hint::PerformanceMode::LATENCY;
    latencyValue = tmp.str();
    tmp.clear();
    tmp << ov::hint::PerformanceMode::THROUGHPUT;
    throughputValue = tmp.str();

    if (value == latencyValue || value == throughputValue || value == "") {
        return value;
    } else {
        VPU_THROW_EXCEPTION << "Wrong value for property key " << CONFIG_KEY(PERFORMANCE_HINT) << ". Expected only "
                            << CONFIG_VALUE(LATENCY) << "/" << CONFIG_VALUE(THROUGHPUT);
    }
}

}  // namespace vpu
