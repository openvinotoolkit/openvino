// Copyright (C) 2018-2025 Intel Corporation
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
extern const char* refCachePath;

extern std::vector<std::string> IRFolderPaths;
extern std::vector<std::string> disabledTests;

enum ShapeMode {
    DYNAMIC,
    STATIC,
    BOTH
};

extern ShapeMode shapeMode;

inline ov::AnyMap read_plugin_config(const std::string& config_file_path) {
    if (!ov::test::utils::fileExists(config_file_path)) {
        OPENVINO_THROW("Input directory (" + config_file_path + ") doesn't not exist!");
    }
    ov::AnyMap config;
    std::ifstream file(config_file_path);
    if (file.is_open()) {
        std::string buffer;
        while (getline(file, buffer)) {
            if (buffer.find("#") == std::string::npos && !buffer.empty()) {
                auto config_elem = ov::test::utils::splitStringByDelimiter(buffer, " ");
                if (config_elem.size() != 2) {
                    OPENVINO_THROW("Incorrect line to get config item: " + buffer +
                                   "\n. Example: \"PLUGIN_CONFIG_KEY=PLUGIN_CONFIG_VALUE\"");
                }
                config.emplace(config_elem.front(), config_elem.back());
            }
        }
    } else {
        OPENVINO_THROW("Error in opening file: " + config_file_path);
    }
    file.close();
    return config;
}

}  // namespace conformance
}  // namespace test
}  // namespace ov
