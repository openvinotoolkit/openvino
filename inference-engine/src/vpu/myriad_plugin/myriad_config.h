// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <unordered_set>

#include <vpu/parsed_config.hpp>

#include <mvnc.h>

namespace vpu {
namespace MyriadPlugin {

struct MyriadConfig final : ParsedConfig {
    ncDevicePlatform_t platform;
    int  watchdogInterval = 1000;
    explicit MyriadConfig(const std::map<std::string, std::string> &config = std::map<std::string, std::string>(),
                          ConfigMode mode = ConfigMode::DEFAULT_MODE);

private:
    std::unordered_set<std::string> getRuntimeOptions() const final;
    void checkInvalidValues(const std::map<std::string, std::string> &config) const final;
};

}  // namespace MyriadPlugin
}  // namespace vpu
