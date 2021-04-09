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
    static const std::unordered_set<std::string> options = ParsedConfig::getCompileOptions();
IE_SUPPRESS_DEPRECATED_END

    return options;
}

const std::unordered_set<std::string>& MyriadConfig::getRunTimeOptions() const {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_set<std::string> options = merge(ParsedConfig::getRunTimeOptions(), {
        ie::MYRIAD_PLUGIN_LOG_FILE_PATH,
    });
IE_SUPPRESS_DEPRECATED_END

    return options;
}

const std::unordered_set<std::string>& MyriadConfig::getDeprecatedOptions() const {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_set<std::string> options = ParsedConfig::getDeprecatedOptions();
IE_SUPPRESS_DEPRECATED_END

    return options;
}

void MyriadConfig::parse(const std::map<std::string, std::string>& config) {
    ParsedConfig::parse(config);

    setOption(_pluginLogFilePath,                       config, ie::MYRIAD_PLUGIN_LOG_FILE_PATH);
    setOption(_enableAsyncDma,   switches,              config, ie::MYRIAD_ENABLE_ASYNC_DMA);

#ifndef NDEBUG
    if (const auto envVar = std::getenv("IE_VPU_MYRIAD_PLUGIN_LOG_FILE_PATH")) {
        _pluginLogFilePath = envVar;
    }
#endif
}

}  // namespace MyriadPlugin
}  // namespace vpu
