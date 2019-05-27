// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vpu/vpu_plugin_config.hpp>
#include <cpp_interfaces/exception2status.hpp>

#include "myriad_config.h"

using namespace vpu;
using namespace vpu::MyriadPlugin;

MyriadConfig::MyriadConfig(const std::map<std::string, std::string> &config, ConfigMode mode) : ParsedConfig(mode)  {
    configure(parse(config));

    platform = UNKNOWN_PLATFORM;
    const std::unordered_map<std::string, ncDevicePlatform_t> platforms = {
        { VPU_CONFIG_VALUE(2450), MYRIAD_2 },
        { VPU_CONFIG_VALUE(2480), MYRIAD_X }
    };

    setOption(platform, platforms, config, VPU_CONFIG_KEY(PLATFORM));

    const std::unordered_map<std::string, int> switches = {
        { CONFIG_VALUE(YES), 1000 },
        { CONFIG_VALUE(NO), 0 }
    };

    setOption(watchdogInterval, switches, config, VPU_CONFIG_KEY(WATCHDOG));

#ifndef NDEBUG
    if (auto envVar = std::getenv("IE_VPU_WATCHDOG_INTERVAL")) {
        watchdogInterval = std::stoi(envVar);
    }
#endif
}

void MyriadConfig::checkInvalidValues(const std::map<std::string, std::string> &config) const {
    ParsedConfig::checkInvalidValues(config);
    checkSupportedValues({{VPU_CONFIG_KEY(PLATFORM), {VPU_CONFIG_VALUE(2450), VPU_CONFIG_VALUE(2480)}}}, config);
}

std::unordered_set<std::string> MyriadConfig::getRuntimeOptions() const {
    auto runtimeOptions = ParsedConfig::getRuntimeOptions();
    runtimeOptions.insert({VPU_CONFIG_KEY(PLATFORM)});
    runtimeOptions.insert({VPU_CONFIG_KEY(WATCHDOG)});
    return runtimeOptions;
}
