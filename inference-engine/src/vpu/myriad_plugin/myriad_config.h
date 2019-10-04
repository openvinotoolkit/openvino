// Copyright (C) 2018-2019 Intel Corporation
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

VPU_DECLARE_ENUM(PowerConfig,
    FULL         = 0,
    INFER        = 1,
    STAGE        = 2,
    STAGE_SHAVES = 3,
    STAGE_NCES   = 4,
)

struct MyriadConfig final : ParsedConfig {
    bool forceReset = false;
    PowerConfig powerConfig = PowerConfig::FULL;
    ncDevicePlatform_t platform = NC_ANY_PLATFORM;
    ncDeviceProtocol_t protocol = NC_ANY_PROTOCOL;
    std::chrono::milliseconds watchdogInterval = std::chrono::milliseconds(1000);
    static constexpr int UNDEFINED_THROUGHPUT_STREAMS = -1;
    int numExecutors = UNDEFINED_THROUGHPUT_STREAMS;
    std::string deviceName;

    explicit MyriadConfig(const std::map<std::string, std::string> &config = std::map<std::string, std::string>(),
                          ConfigMode mode = ConfigMode::DEFAULT_MODE);

private:
    std::unordered_set<std::string> getRuntimeOptions() const final;
    void checkInvalidValues(const std::map<std::string, std::string> &config) const final;
};

}  // namespace MyriadPlugin
}  // namespace vpu
