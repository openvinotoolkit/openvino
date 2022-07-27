// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/file_utils.hpp"

namespace ov {
namespace test {
namespace conformance {
extern const char* targetDevice;
extern const char *targetPluginName;

extern std::vector<std::string> IRFolderPaths;
extern std::vector<std::string> disabledTests;

extern ov::AnyMap pluginConfig;

inline ov::AnyMap readPluginConfig(const std::string &configFilePath) {
    if (!CommonTestUtils::fileExists(configFilePath)) {
        std::string msg = "Input directory (" + configFilePath + ") doesn't not exist!";
        throw std::runtime_error(msg);
    }
    ov::AnyMap config;
    std::ifstream file(configFilePath);
    if (file.is_open()) {
        std::string buffer;
        while (getline(file, buffer)) {
            if (buffer.find("#") == std::string::npos && !buffer.empty()) {
                auto configElements = CommonTestUtils::splitStringByDelimiter(buffer, " ");
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

inline std::vector<std::string> getModelPaths(const std::vector<std::string>& conformance_ir_paths) {
    std::vector<std::string> result;
    for (const auto& conformance_ir_path : conformance_ir_paths) {
        std::vector<std::string> tmp_buf;
        if (CommonTestUtils::directoryExists(conformance_ir_path)) {
            tmp_buf = CommonTestUtils::getFileListByPatternRecursive({conformance_ir_path}, {std::regex(R"(.*\.xml)")});
        } else if (CommonTestUtils::fileExists(conformance_ir_path)) {
            tmp_buf = CommonTestUtils::readListFiles({conformance_ir_path});
        } else {
            continue;
        }
        result.insert(result.end(), tmp_buf.begin(), tmp_buf.end());
    }
    return result;
}

}  // namespace conformance
}  // namespace test
}  // namespace ov
