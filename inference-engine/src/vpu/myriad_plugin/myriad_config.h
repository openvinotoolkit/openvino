// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <unordered_set>
#include <chrono>

#include <vpu/parsed_config.hpp>

#include <mvnc.h>

namespace vpu {
namespace MyriadPlugin {

// Must be synchronized with firmware side.
VPU_DECLARE_ENUM(PowerConfig,
    FULL         = 0,
    INFER        = 1,
    STAGE        = 2,
    STAGE_SHAVES = 3,
    STAGE_NCES   = 4,
)

class MyriadConfig final : public ParsedConfig {
public:
    const std::string& pluginLogFilePath() const {
        return _pluginLogFilePath;
    }

    bool forceReset() const {
        return _forceReset;
    }

    PowerConfig powerConfig() const {
        return _powerConfig;
    }

    ncDevicePlatform_t platform() const {
        return _platform;
    }

    ncDeviceProtocol_t protocol() const {
        return _protocol;
    }

    const std::chrono::milliseconds& watchdogInterval() const {
        return _watchdogInterval;
    }

    const std::chrono::seconds& deviceConnectTimeout() const {
        return _deviceConnectTimeout;
    }

    const std::string& deviceName() const {
        return _deviceName;
    }

protected:
    const std::unordered_set<std::string>& getCompileOptions() const override;
    const std::unordered_set<std::string>& getRunTimeOptions() const override;
    const std::unordered_set<std::string>& getDeprecatedOptions() const override;
    void parse(const std::map<std::string, std::string>& config) override;

private:
    std::string _pluginLogFilePath;
    bool _forceReset = false;
    PowerConfig _powerConfig = PowerConfig::FULL;
    ncDevicePlatform_t _platform = NC_ANY_PLATFORM;
    ncDeviceProtocol_t _protocol = NC_ANY_PROTOCOL;
    std::chrono::milliseconds _watchdogInterval = std::chrono::milliseconds(1000);
    std::chrono::seconds _deviceConnectTimeout = std::chrono::seconds(15);
    std::string _deviceName;
};

}  // namespace MyriadPlugin
}  // namespace vpu
