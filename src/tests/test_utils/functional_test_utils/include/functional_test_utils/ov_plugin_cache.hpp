// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "openvino/runtime/core.hpp"

#include "common_test_utils/file_utils.hpp"
#include "openvino/util/file_util.hpp"

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
    std::shared_ptr<ov::Core> ov_core;
};

extern ov::AnyMap pluginConfig;

inline ov::AnyMap readPluginConfig(const std::string &configFilePath) {
    if (!ov::test::utils::fileExists(configFilePath)) {
        std::string msg = "Input directory (" + configFilePath + ") doesn't not exist!";
        throw std::runtime_error(msg);
    }
    ov::AnyMap config;
    std::ifstream file(configFilePath);
    if (file.is_open()) {
        std::string buffer;
        while (getline(file, buffer)) {
            if (buffer.find("#") == std::string::npos && !buffer.empty()) {
                auto configElements = ov::test::utils::splitStringByDelimiter(buffer, " ");
                if (configElements.size() != 2) {
                    throw std::runtime_error("Incorrect line to get config item: " + buffer + "\n. Example: \"PLUGIN_CONFIG_KEY=PLUGIN_CONFIG_VALUE\"");
                }
                config.emplace(configElements.front(), configElements.back());
            }
        }
    } else {
        std::string msg = "Error in opening file: " + configFilePath;
        throw std::runtime_error(msg);
    }
    file.close();
    return config;
}
}  // namespace utils
}  // namespace test
}  // namespace ov
