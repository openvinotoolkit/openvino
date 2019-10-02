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

namespace vpu {
namespace MyriadPlugin {

const std::unordered_set<std::string>& MyriadConfig::getCompileOptions() const {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_set<std::string> options = merge(ParsedConfig::getCompileOptions(), {
        VPU_MYRIAD_CONFIG_KEY(PLATFORM),
        VPU_CONFIG_KEY(PLATFORM),
    });
IE_SUPPRESS_DEPRECATED_END

    return options;
}

const std::unordered_set<std::string>& MyriadConfig::getRunTimeOptions() const {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_set<std::string> options = merge(ParsedConfig::getRunTimeOptions(), {
        CONFIG_KEY(DEVICE_ID),

        VPU_MYRIAD_CONFIG_KEY(FORCE_RESET),
        VPU_MYRIAD_CONFIG_KEY(PLATFORM),
        VPU_MYRIAD_CONFIG_KEY(PROTOCOL),
        VPU_MYRIAD_CONFIG_KEY(WATCHDOG),
        VPU_MYRIAD_CONFIG_KEY(THROUGHPUT_STREAMS),
        VPU_MYRIAD_CONFIG_KEY(POWER_MANAGEMENT),

        VPU_CONFIG_KEY(FORCE_RESET),
        VPU_CONFIG_KEY(PLATFORM),
    });
IE_SUPPRESS_DEPRECATED_END

    return options;
}

void MyriadConfig::parse(const std::map<std::string, std::string>& config) {
    static const std::unordered_map<std::string, ncDevicePlatform_t> platforms = {
        { VPU_MYRIAD_CONFIG_VALUE(2450),   NC_MYRIAD_2 },
        { VPU_MYRIAD_CONFIG_VALUE(2480),   NC_MYRIAD_X },
        { std::string(),                   NC_ANY_PLATFORM }
    };

IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_map<std::string, ncDevicePlatform_t> platformsDepr = {
        { VPU_CONFIG_VALUE(2450), NC_MYRIAD_2 },
        { VPU_CONFIG_VALUE(2480), NC_MYRIAD_X },
        { std::string(),          NC_ANY_PLATFORM }
    };
IE_SUPPRESS_DEPRECATED_END

    static const std::unordered_map<std::string, ncDeviceProtocol_t> protocols = {
        { VPU_MYRIAD_CONFIG_VALUE(USB),     NC_USB},
        { VPU_MYRIAD_CONFIG_VALUE(PCIE),    NC_PCIE},
        { std::string(),                    NC_ANY_PROTOCOL}
    };

    static const std::unordered_map<std::string, std::chrono::milliseconds> watchdogIntervals = {
        { CONFIG_VALUE(YES), std::chrono::milliseconds(1000) },
        { CONFIG_VALUE(NO), std::chrono::milliseconds(0) }
    };

    static const std::unordered_map<std::string, PowerConfig> powerConfigs = {
        { VPU_MYRIAD_CONFIG_VALUE(POWER_FULL),         PowerConfig::FULL },
        { VPU_MYRIAD_CONFIG_VALUE(POWER_INFER),        PowerConfig::INFER },
        { VPU_MYRIAD_CONFIG_VALUE(POWER_STAGE),        PowerConfig::STAGE },
        { VPU_MYRIAD_CONFIG_VALUE(POWER_STAGE_SHAVES), PowerConfig::STAGE_SHAVES },
        { VPU_MYRIAD_CONFIG_VALUE(POWER_STAGE_NCES),   PowerConfig::STAGE_NCES },
    };

    ParsedConfig::parse(config);

    setOption(_deviceName, config, CONFIG_KEY(DEVICE_ID));
    setOption(_forceReset, switches, config, VPU_MYRIAD_CONFIG_KEY(FORCE_RESET));
    setOption(_platform, platforms, config, VPU_MYRIAD_CONFIG_KEY(PLATFORM));
    setOption(_protocol, protocols, config, VPU_MYRIAD_CONFIG_KEY(PROTOCOL));
    setOption(_watchdogInterval, watchdogIntervals, config, VPU_MYRIAD_CONFIG_KEY(WATCHDOG));
    setOption(_powerConfig, powerConfigs, config, VPU_MYRIAD_CONFIG_KEY(POWER_MANAGEMENT));
    setOption(_numExecutors, config, VPU_MYRIAD_CONFIG_KEY(THROUGHPUT_STREAMS), parseInt);

IE_SUPPRESS_DEPRECATED_START
    setOption(_forceReset, switches, config, VPU_CONFIG_KEY(FORCE_RESET));
    setOption(_platform, platformsDepr, config, VPU_CONFIG_KEY(PLATFORM));
IE_SUPPRESS_DEPRECATED_END

#ifndef NDEBUG
    if (const auto envVar = std::getenv("IE_VPU_MYRIAD_FORCE_RESET")) {
        _forceReset = std::stoi(envVar);
    }
    if (const auto envVar = std::getenv("IE_VPU_MYRIAD_WATCHDOG_INTERVAL")) {
        _watchdogInterval = std::chrono::milliseconds(std::stoi(envVar));
    }
#endif
}

}  // namespace MyriadPlugin
}  // namespace vpu
