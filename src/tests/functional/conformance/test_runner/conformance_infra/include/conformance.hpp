// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include <list>
#include <algorithm>

#include "openvino/opsets/opset.hpp"

#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/node_utils.hpp"

namespace ov {
namespace test {
namespace conformance {
extern const char* targetDevice;
extern const char *targetPluginName;
extern const char* refCachePath;

extern std::vector<std::string> IRFolderPaths;
extern std::vector<std::string> disabledTests;

extern ov::AnyMap pluginConfig;

enum ShapeMode {
    DYNAMIC,
    STATIC,
    BOTH
};

extern ShapeMode shapeMode;

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

}  // namespace conformance
}  // namespace test
}  // namespace ov
