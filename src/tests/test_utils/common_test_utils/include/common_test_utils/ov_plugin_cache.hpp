// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "common_test_utils/file_utils.hpp"
#include "openvino/runtime/core.hpp"

namespace ov {
namespace test {
namespace utils {

ov::Core create_core(const std::string& target_device = "");

class PluginCache {
public:
    std::shared_ptr<ov::Core> core(const std::string& deviceToCheck = std::string());

    static PluginCache& get();

    void reset();

    PluginCache(const PluginCache&) = delete;

    PluginCache& operator=(const PluginCache&) = delete;

private:
    PluginCache();

    ~PluginCache() = default;

    std::mutex g_mtx;
    bool disable_plugin_cache;
    std::shared_ptr<ov::Core> ov_core;
};

extern ov::AnyMap pluginConfig;

inline ov::AnyMap readPluginConfig(const std::string& configFilePath) {
    if (!ov::test::utils::fileExists(configFilePath)) {
        OPENVINO_THROW("Input directory (" + configFilePath + ") doesn't not exist!");
    }
    ov::AnyMap config;
    std::ifstream file(configFilePath);
    if (file.is_open()) {
        std::string buffer;
        while (getline(file, buffer)) {
            if (buffer.find("#") == std::string::npos && !buffer.empty()) {
                auto configElements = ov::test::utils::splitStringByDelimiter(buffer, " ");
                if (configElements.size() != 2) {
                    OPENVINO_THROW("Incorrect line to get config item: " + buffer +
                                   "\n. Example: \"PLUGIN_CONFIG_KEY=PLUGIN_CONFIG_VALUE\"");
                }
                config.emplace(configElements.front(), configElements.back());
            }
        }
    } else {
        OPENVINO_THROW("Error in opening file: " + configFilePath);
    }
    file.close();
    return config;
};
}  // namespace utils
}  // namespace test
}  // namespace ov
