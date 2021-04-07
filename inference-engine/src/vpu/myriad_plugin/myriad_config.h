// Copyright (C) 2018-2021 Intel Corporation
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

class MyriadConfig : public virtual ParsedConfig {
public:
    const std::string& pluginLogFilePath() const {
        return _pluginLogFilePath;
    }

    bool forceReset() const {
        return _forceReset;
    }

    bool asyncDma() const {
        return _enableAsyncDma;
    }

    ncDevicePlatform_t platform() const {
        return _platform;
    }

protected:
    const std::unordered_set<std::string>& getCompileOptions() const override;
    const std::unordered_set<std::string>& getRunTimeOptions() const override;
    const std::unordered_set<std::string>& getDeprecatedOptions() const override;
    void parse(const std::map<std::string, std::string>& config) override;

private:
    std::string _pluginLogFilePath;
    bool _forceReset = false;
    bool _enableAsyncDma = true;
    ncDevicePlatform_t _platform = NC_ANY_PLATFORM;
};

}  // namespace MyriadPlugin
}  // namespace vpu
