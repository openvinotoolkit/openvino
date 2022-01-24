// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common_test_utils/file_utils.hpp>

namespace ConformanceTests {

extern const char* targetDevice;
extern const char* targetPluginName;
extern std::vector<std::string> IRFolderPaths;
extern std::vector<std::string> disabledTests;
extern std::map<std::string, std::string> pluginConfig;

inline std::map<std::string, std::string> readPluginConfig(const std::string& configFilePath) {
    if (!CommonTestUtils::fileExists(configFilePath)) {
        std::string msg = "Input directory (" + configFilePath + ") doesn't not exist!";
        throw std::runtime_error(msg);
    }
    std::map<std::string, std::string> config;
    std::ifstream file(configFilePath);
    if (file.is_open()) {
        std::string buffer;
        while (getline(file, buffer)) {
            if (buffer.find("#") == std::string::npos && !buffer.empty()) {
                auto configElements = CommonTestUtils::splitStringByDelimiter(buffer, " ");
                if (configElements.size() != 2) {
                    throw std::runtime_error("Incorrect line to get config item: " + buffer + "\n. Example: \"PLUGIN_CONFIG_KEY=PLUGIN_CONFIG_VALUE\"");
                }
                std::pair<std::string, std::string> configItem{configElements.front(), configElements.back()};
                config.insert(configItem);
            }
        }
    } else {
        std::string msg = "Error in opening file: " + configFilePath;
        throw std::runtime_error(msg);
    }
    file.close();
    return config;
}

} // namespace ConformanceTests
