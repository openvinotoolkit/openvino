// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/perf_report_mode.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

#include <unordered_map>

namespace vpu {

namespace {

const std::unordered_map<std::string, PerfReport>& string2mode() {
    static const std::unordered_map<std::string, PerfReport> converters = {
        {InferenceEngine::MYRIAD_PER_LAYER, PerfReport::PerLayer},
        {InferenceEngine::MYRIAD_PER_STAGE, PerfReport::PerStage}
    };
    return converters;
}

}  // namespace

void PerfReportModeOption::validate(const std::string& value) {
    const auto& converters = string2mode();
    VPU_THROW_UNLESS(converters.count(value) != 0,
        R"(unexpected {} option value "{}", only {} are supported)", key(), value, getKeys(converters));
}

void PerfReportModeOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string PerfReportModeOption::key() {
    return InferenceEngine::MYRIAD_PERF_REPORT_MODE;
}

details::Access PerfReportModeOption::access() {
    return details::Access::Private;
}

details::Category PerfReportModeOption::category() {
    return details::Category::RunTime;
}

std::string PerfReportModeOption::defaultValue() {
    return InferenceEngine::MYRIAD_PER_LAYER;
}

PerfReportModeOption::value_type PerfReportModeOption::parse(const std::string& value) {
    const auto& converters = string2mode();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0,
        R"(unexpected {} option value "{}", only {} are supported)", key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
