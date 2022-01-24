// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/configuration/options/log_level.hpp"
#include "vpu/utils/log_level.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

#include "ie_plugin_config.hpp"

#include <unordered_map>

namespace vpu {

namespace {

const std::unordered_map<std::string, LogLevel>& string2level() {
    static const std::unordered_map<std::string, LogLevel> converters = {
        {CONFIG_VALUE(LOG_NONE),    LogLevel::None},
        {CONFIG_VALUE(LOG_ERROR),   LogLevel::Error},
        {CONFIG_VALUE(LOG_WARNING), LogLevel::Warning},
        {CONFIG_VALUE(LOG_INFO),    LogLevel::Info},
        {CONFIG_VALUE(LOG_DEBUG),   LogLevel::Debug},
        {CONFIG_VALUE(LOG_TRACE),   LogLevel::Trace},
    };
    return converters;
}

}  // namespace

void LogLevelOption::validate(const std::string& value) {
    const auto& converters = string2level();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected log level option value "{}", only {} are supported)", value, getKeys(converters));
}

void LogLevelOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string LogLevelOption::key() {
    return InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL;
}

details::Access LogLevelOption::access() {
    return details::Access::Public;
}

details::Category LogLevelOption::category() {
    return details::Category::CompileTime;
}

std::string LogLevelOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::LOG_NONE;
}

LogLevelOption::value_type LogLevelOption::parse(const std::string& value) {
    const auto& converters = string2level();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected log level option value "{}", only {} are supported)",
        value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
