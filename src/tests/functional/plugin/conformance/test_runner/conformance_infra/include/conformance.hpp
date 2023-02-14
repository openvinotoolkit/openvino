// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/file_utils.hpp"
#include <list>

namespace ov {
namespace test {
namespace conformance {
extern const char* targetDevice;
extern const char *targetPluginName;

extern std::vector<std::string> IRFolderPaths;
extern std::vector<std::string> disabledTests;
extern std::list<std::string> dirList;

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

inline std::vector<std::string> getModelPaths(const std::vector<std::string>& conformance_ir_paths,
                                              const std::string opName = "Other") {
    // This is required to prevent re-scan folders each call in case there is nothing found
    static bool listPrepared = false;
    if (!listPrepared) {
        // Looking for any applicable files in a folders
        for (const auto& conformance_ir_path : conformance_ir_paths) {
            std::vector<std::string> tmp_buf;
            if (CommonTestUtils::directoryExists(conformance_ir_path)) {
                tmp_buf =
                    CommonTestUtils::getFileListByPatternRecursive({conformance_ir_path}, {std::regex(R"(.*\.xml)")});
            } else if (CommonTestUtils::fileExists(conformance_ir_path)) {
                tmp_buf = CommonTestUtils::readListFiles({conformance_ir_path});
            } else {
                continue;
            }
            //Save it in a list
            dirList.insert(dirList.end(), tmp_buf.begin(), tmp_buf.end());
        }
        listPrepared = true;
    }

    std::vector<std::string> result;
    if (!opName.empty() && opName != "Other") {
        std::string strToFind = CommonTestUtils::FileSeparator + opName + CommonTestUtils::FileSeparator;
        auto it = dirList.begin();
        while (it != dirList.end()) {
            if (it->find(strToFind) != std::string::npos) {
                result.push_back(*it);
                it = dirList.erase(it);
            } else {
                ++it;
            }
        }
    } else if (opName == "Other") {
        // For "Undefined" operation name - run all applicable files in "Undefined" handler
        result.insert(result.end(), dirList.begin(), dirList.end());
    } else {
        std::string message = "Operatiion name: " + opName + " is incorrect. Please check the instantiation parameters!";
        throw std::runtime_error(message);
    }
    return result;
}

}  // namespace conformance
}  // namespace test
}  // namespace ov
