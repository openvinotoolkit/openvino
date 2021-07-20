// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/watchdog_interval.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/utils/error.hpp"

#include <unordered_map>

namespace vpu {

namespace {

const std::unordered_map<std::string, std::chrono::milliseconds>& string2interval() {
    static const std::unordered_map<std::string, std::chrono::milliseconds> converters = {
        {CONFIG_VALUE(NO), std::chrono::milliseconds(0)},
        {CONFIG_VALUE(YES), std::chrono::milliseconds(1000)}
    };
    return converters;
}

}  // namespace

void WatchdogIntervalOption::validate(const std::string& value) {
    const auto& converters = string2interval();

#ifndef NDEBUG
    if (converters.count(value) == 0) {
        int intValue;
        try {
            intValue = std::stoi(value);
        } catch (const std::exception& e) {
            VPU_THROW_FORMAT(R"(unexpected {} option value "{}", must be a number)", key(), value);
        }

        VPU_THROW_UNSUPPORTED_OPTION_UNLESS(intValue >= 0,
            R"(unexpected {} option value "{}", only {} and not negative numbers are supported)", key(), value, getKeys(converters));
        return;
    }
#endif

    VPU_THROW_UNLESS(converters.count(value) != 0,
        R"(unexpected {} option value "{}", only {} are supported)", key(), value, getKeys(converters));
}

void WatchdogIntervalOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string WatchdogIntervalOption::key() {
    return InferenceEngine::MYRIAD_WATCHDOG;
}

details::Access WatchdogIntervalOption::access() {
    return details::Access::Private;
}

details::Category WatchdogIntervalOption::category() {
    return details::Category::RunTime;
}

std::string WatchdogIntervalOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::YES;
}

WatchdogIntervalOption::value_type WatchdogIntervalOption::parse(const std::string& value) {
    const auto& converters = string2interval();

#ifndef NDEBUG
    if (converters.count(value) == 0) {
        int intValue;
        try {
            intValue = std::stoi(value);
        } catch (const std::exception& e) {
            VPU_THROW_FORMAT(R"(unexpected {} option value "{}", must be a number)", key(), value);
        }

        VPU_THROW_UNSUPPORTED_OPTION_UNLESS(intValue >= 0,
            R"(unexpected {} option value "{}", only {} and not negative numbers are supported)", key(), value, getKeys(converters));
        return WatchdogIntervalOption::value_type(intValue);
    }
#endif

    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0,
        R"(unexpected {} option value "{}", only {} are supported)", key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
