// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_config.h"

#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

#include <cpp_interfaces/exception2status.hpp>

#include <vpu/vpu_plugin_config.hpp>

using namespace vpu;
using namespace vpu::MyriadPlugin;


MyriadConfig::MyriadConfig(const std::map<std::string, std::string> &config, ConfigMode mode) : ParsedConfig(mode)  {
    configure(parse(config));

    static const std::unordered_map<std::string, bool> boolSwitches = {
        { CONFIG_VALUE(YES), true },
        { CONFIG_VALUE(NO), false }
    };
    static const std::unordered_map<std::string, ncDevicePlatform_t> platformSwitches = {
        { VPU_MYRIAD_CONFIG_VALUE(2450),   NC_MYRIAD_2 },
        { VPU_MYRIAD_CONFIG_VALUE(2480),   NC_MYRIAD_X },
        { std::string(),                   NC_ANY_PLATFORM }
    };
    static const std::unordered_map<std::string, ncDeviceProtocol_t> protocolSwitches = {
        { VPU_MYRIAD_CONFIG_VALUE(USB),     NC_USB},
        { VPU_MYRIAD_CONFIG_VALUE(PCIE),    NC_PCIE},
        { std::string(),                    NC_ANY_PROTOCOL}
    };
    static const std::unordered_map<std::string, std::chrono::milliseconds> watchdogSwitches = {
        { CONFIG_VALUE(YES), std::chrono::milliseconds(1000) },
        { CONFIG_VALUE(NO), std::chrono::milliseconds(0) }
    };

    setOption(forceReset, boolSwitches, config, VPU_MYRIAD_CONFIG_KEY(FORCE_RESET));
    setOption(platform, platformSwitches, config, VPU_MYRIAD_CONFIG_KEY(PLATFORM));
    setOption(protocol, protocolSwitches, config, VPU_MYRIAD_CONFIG_KEY(PROTOCOL));
    setOption(watchdogInterval, watchdogSwitches, config, VPU_MYRIAD_CONFIG_KEY(WATCHDOG));

IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_map<std::string, ncDevicePlatform_t> platformSwitchesDepr = {
        { VPU_CONFIG_VALUE(2450), NC_MYRIAD_2 },
        { VPU_CONFIG_VALUE(2480), NC_MYRIAD_X },
        { std::string(),          NC_ANY_PLATFORM }
    };

    setOption(forceReset, boolSwitches, config, VPU_CONFIG_KEY(FORCE_RESET));
    setOption(platform, platformSwitchesDepr, config, VPU_CONFIG_KEY(PLATFORM));
    setOption(watchdogInterval, watchdogSwitches, config, VPU_CONFIG_KEY(WATCHDOG));
IE_SUPPRESS_DEPRECATED_END

#ifndef NDEBUG
    if (auto envVar = std::getenv("IE_VPU_MYRIAD_FORCE_RESET")) {
        forceReset = std::stoi(envVar);
    }
    if (auto envVar = std::getenv("IE_VPU_MYRIAD_WATCHDOG_INTERVAL")) {
        watchdogInterval = std::chrono::milliseconds(std::stoi(envVar));
    }
#endif

    setOption(
        numExecutors, config, VPU_MYRIAD_CONFIG_KEY(THROUGHPUT_STREAMS),
        [](const std::string& src) { return std::stoi(src); });

    setOption(deviceName, config, CONFIG_KEY(DEVICE_ID));
}

void MyriadConfig::checkInvalidValues(const std::map<std::string, std::string> &config) const {
    ParsedConfig::checkInvalidValues(config);

IE_SUPPRESS_DEPRECATED_START
    checkSupportedValues({
        {VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), {CONFIG_VALUE(YES), CONFIG_VALUE(NO)}},
        {VPU_MYRIAD_CONFIG_KEY(PLATFORM),
                { VPU_MYRIAD_CONFIG_VALUE(2450), VPU_MYRIAD_CONFIG_VALUE(2480), std::string()}},
        {VPU_MYRIAD_CONFIG_KEY(PROTOCOL),
                { VPU_MYRIAD_CONFIG_VALUE(PCIE), VPU_MYRIAD_CONFIG_VALUE(USB), std::string()}},
        {VPU_MYRIAD_CONFIG_KEY(WATCHDOG),    {CONFIG_VALUE(YES), CONFIG_VALUE(NO)}},

        {VPU_CONFIG_KEY(FORCE_RESET),        {CONFIG_VALUE(YES), CONFIG_VALUE(NO)}},
        {VPU_CONFIG_KEY(PLATFORM),
                { VPU_CONFIG_VALUE(2450), VPU_CONFIG_VALUE(2480), std::string()}},
        {VPU_CONFIG_KEY(WATCHDOG),           {CONFIG_VALUE(YES), CONFIG_VALUE(NO)}}
    }, config);

    auto throughput_streams = config.find(VPU_MYRIAD_CONFIG_KEY(THROUGHPUT_STREAMS));
    if (throughput_streams != config.end()) {
        try {
            std::stoi(throughput_streams->second);
        }
        catch(...) {
            THROW_IE_EXCEPTION << "Invalid config value for VPU_MYRIAD_THROUGHPUT_STREAMS, can't cast to int";
        }
    }

    if (config.find(VPU_CONFIG_KEY(FORCE_RESET)) != config.end() &&
        config.find(VPU_MYRIAD_CONFIG_KEY(FORCE_RESET)) != config.end()) {
        THROW_IE_EXCEPTION << "VPU_MYRIAD_FORCE_RESET and VPU_FORCE_RESET cannot be set simultaneously.";
    }

    if (config.find(VPU_CONFIG_KEY(PLATFORM)) != config.end() &&
        config.find(VPU_MYRIAD_CONFIG_KEY(PLATFORM)) != config.end()) {
        THROW_IE_EXCEPTION << "VPU_MYRIAD_PLATFORM and VPU_PLATFORM cannot be set simultaneously.";
    }

    if (config.find(VPU_CONFIG_KEY(WATCHDOG)) != config.end() &&
        config.find(VPU_MYRIAD_CONFIG_KEY(WATCHDOG)) != config.end()) {
        THROW_IE_EXCEPTION << "VPU_MYRIAD_WATCHDOG and VPU_WATCHDOG cannot be set simultaneously.";
    }

IE_SUPPRESS_DEPRECATED_END
}

std::unordered_set<std::string> MyriadConfig::getRuntimeOptions() const {
    auto runtimeOptions = ParsedConfig::getRuntimeOptions();

IE_SUPPRESS_DEPRECATED_START
    runtimeOptions.insert({
        {VPU_MYRIAD_CONFIG_KEY(FORCE_RESET)},
        {VPU_MYRIAD_CONFIG_KEY(PLATFORM)},
        {VPU_MYRIAD_CONFIG_KEY(PROTOCOL)},
        {VPU_MYRIAD_CONFIG_KEY(WATCHDOG)},
        {VPU_MYRIAD_CONFIG_KEY(THROUGHPUT_STREAMS)},

        {VPU_CONFIG_KEY(FORCE_RESET)},
        {VPU_CONFIG_KEY(PLATFORM)},
        {VPU_CONFIG_KEY(WATCHDOG)},

        {CONFIG_KEY(DEVICE_ID)}
    });
IE_SUPPRESS_DEPRECATED_END

    return runtimeOptions;
}
