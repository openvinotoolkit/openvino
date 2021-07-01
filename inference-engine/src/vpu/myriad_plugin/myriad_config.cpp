// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_config.h"

#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include <vpu/vpu_plugin_config.hpp>
#include <vpu/myriad_config.hpp>

namespace vpu {
namespace MyriadPlugin {

const std::unordered_set<std::string>& MyriadConfig::getCompileOptions() const {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_set<std::string> options = merge(ParsedConfig::getCompileOptions(), {
        VPU_MYRIAD_CONFIG_KEY(PLATFORM),
    });
IE_SUPPRESS_DEPRECATED_END

    return options;
}

const std::unordered_set<std::string>& MyriadConfig::getRunTimeOptions() const {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_set<std::string> options = merge(ParsedConfig::getRunTimeOptions(), {
        CONFIG_KEY(DEVICE_ID),

        ie::MYRIAD_ENABLE_FORCE_RESET,

        ie::MYRIAD_PROTOCOL,
        ie::MYRIAD_WATCHDOG,
        ie::MYRIAD_THROUGHPUT_STREAMS,
        ie::MYRIAD_POWER_MANAGEMENT,

        ie::MYRIAD_PLUGIN_LOG_FILE_PATH,
        ie::MYRIAD_DEVICE_CONNECT_TIMEOUT,

        ie::MYRIAD_DDR_TYPE,

        // Deprecated
        VPU_MYRIAD_CONFIG_KEY(FORCE_RESET),
        VPU_MYRIAD_CONFIG_KEY(PLATFORM),
        VPU_MYRIAD_CONFIG_KEY(PROTOCOL),
        VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE),
    });
IE_SUPPRESS_DEPRECATED_END

    return options;
}

const std::unordered_set<std::string>& MyriadConfig::getDeprecatedOptions() const {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_set<std::string> options = merge(ParsedConfig::getDeprecatedOptions(), {
        VPU_MYRIAD_CONFIG_KEY(FORCE_RESET),
        VPU_MYRIAD_CONFIG_KEY(PLATFORM),
        VPU_MYRIAD_CONFIG_KEY(PROTOCOL),
        VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE),
    });
IE_SUPPRESS_DEPRECATED_END

    return options;
}

void MyriadConfig::parse(const std::map<std::string, std::string>& config) {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_map<std::string, ncDevicePlatform_t> platformsDeprecated = {
        { VPU_MYRIAD_CONFIG_VALUE(2450), NC_MYRIAD_2 },
        { VPU_MYRIAD_CONFIG_VALUE(2480), NC_MYRIAD_X },
        { std::string(),                 NC_ANY_PLATFORM }
    };

    static const std::unordered_map<std::string, ncDeviceProtocol_t> protocolsDeprecated = {
        { VPU_MYRIAD_CONFIG_VALUE(USB),  NC_USB},
        { VPU_MYRIAD_CONFIG_VALUE(PCIE), NC_PCIE},
        { std::string(),                 NC_ANY_PROTOCOL}
    };

    static const std::unordered_map<std::string, MovidiusDdrType> memoryTypesDeprecated = {
         { VPU_MYRIAD_CONFIG_VALUE(DDR_AUTO),     MovidiusDdrType::AUTO },
         { VPU_MYRIAD_CONFIG_VALUE(MICRON_2GB),   MovidiusDdrType::MICRON_2GB },
         { VPU_MYRIAD_CONFIG_VALUE(SAMSUNG_2GB),  MovidiusDdrType::SAMSUNG_2GB },
         { VPU_MYRIAD_CONFIG_VALUE(HYNIX_2GB),    MovidiusDdrType::HYNIX_2GB },
         { VPU_MYRIAD_CONFIG_VALUE(MICRON_1GB),   MovidiusDdrType::MICRON_1GB }
    };
IE_SUPPRESS_DEPRECATED_END

    static const std::unordered_map<std::string, ncDeviceProtocol_t> protocols = {
        { ie::MYRIAD_USB,     NC_USB},
        { ie::MYRIAD_PCIE,    NC_PCIE},
        { std::string(),      NC_ANY_PROTOCOL}
    };

    static const std::unordered_map<std::string, std::chrono::milliseconds> watchdogIntervals = {
        { CONFIG_VALUE(YES), std::chrono::milliseconds(1000) },
        { CONFIG_VALUE(NO), std::chrono::milliseconds(0) }
    };

    static const std::unordered_map<std::string, PowerConfig> powerConfigs = {
        { ie::MYRIAD_POWER_FULL,         PowerConfig::FULL },
        { ie::MYRIAD_POWER_INFER,        PowerConfig::INFER },
        { ie::MYRIAD_POWER_STAGE,        PowerConfig::STAGE },
        { ie::MYRIAD_POWER_STAGE_SHAVES, PowerConfig::STAGE_SHAVES },
        { ie::MYRIAD_POWER_STAGE_NCES,   PowerConfig::STAGE_NCES },
    };

    static const std::unordered_map<std::string, MovidiusDdrType> memoryTypes = {
        { ie::MYRIAD_DDR_AUTO,         MovidiusDdrType::AUTO },
        { ie::MYRIAD_DDR_MICRON_2GB,   MovidiusDdrType::MICRON_2GB },
        { ie::MYRIAD_DDR_SAMSUNG_2GB,  MovidiusDdrType::SAMSUNG_2GB },
        { ie::MYRIAD_DDR_HYNIX_2GB,    MovidiusDdrType::HYNIX_2GB },
        { ie::MYRIAD_DDR_MICRON_1GB,   MovidiusDdrType::MICRON_1GB }
    };

    ParsedConfig::parse(config);

    setOption(_pluginLogFilePath,                       config, ie::MYRIAD_PLUGIN_LOG_FILE_PATH);
    setOption(_deviceName,                              config, CONFIG_KEY(DEVICE_ID));
    setOption(_forceReset,       switches,              config, ie::MYRIAD_ENABLE_FORCE_RESET);
    setOption(_protocol,         protocols,             config, ie::MYRIAD_PROTOCOL);
    setOption(_watchdogInterval, watchdogIntervals,     config, ie::MYRIAD_WATCHDOG);
    setOption(_deviceConnectTimeout,                    config, ie::MYRIAD_DEVICE_CONNECT_TIMEOUT, parseSeconds);
    setOption(_powerConfig,      powerConfigs,          config, ie::MYRIAD_POWER_MANAGEMENT);
    setOption(_memoryType,       memoryTypes,           config, ie::MYRIAD_DDR_TYPE);
    setOption(_enableAsyncDma,   switches,              config, ie::MYRIAD_ENABLE_ASYNC_DMA);

IE_SUPPRESS_DEPRECATED_START
    setOption(_forceReset,       switches,              config, VPU_MYRIAD_CONFIG_KEY(FORCE_RESET));
    setOption(_platform,         platformsDeprecated,   config, VPU_MYRIAD_CONFIG_KEY(PLATFORM));
    setOption(_protocol,         protocolsDeprecated,   config, VPU_MYRIAD_CONFIG_KEY(PROTOCOL));
    setOption(_memoryType,       memoryTypesDeprecated, config, VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE));
IE_SUPPRESS_DEPRECATED_END

#ifndef NDEBUG
    if (const auto envVar = std::getenv("IE_VPU_MYRIAD_PLUGIN_LOG_FILE_PATH")) {
        _pluginLogFilePath = envVar;
    }
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
