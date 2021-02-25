// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <regex>

#include "common_test_utils/file_utils.hpp"

namespace FuncTestUtils {
static std::vector<std::string> getXmlPathsFromFolderRecursive(const std::vector<std::string> &folderPaths) {
    auto getXmlPaths = [](const std::string &folderPath) {
        auto xmlPattern = std::regex(R"(.*\.xml)");
        std::vector<std::string> allFilePaths;
        CommonTestUtils::directoryFileListRecursive(folderPath, allFilePaths);
        std::set<std::string> xmlPaths;
        for (auto& filePath : allFilePaths) {
            if (CommonTestUtils::fileExists(filePath) && std::regex_match(filePath, xmlPattern)) {
                xmlPaths.insert(filePath);
            }
        }
        return xmlPaths;
    };

    std::vector<std::string> xmlPaths;
    for (auto &&folderPath : folderPaths) {
        if (!CommonTestUtils::directoryExists(folderPath)) {
            continue;
        }
        auto xmls = getXmlPaths(folderPath);
        xmlPaths.insert(xmlPaths.end(), xmls.begin(), xmls.end());
    }
    return xmlPaths;
}

static std::vector<std::string> splitString(std::string str) {
    std::string delimiter(",");
    size_t delimiterPos;
    std::vector<std::string> irPaths;
    while ((delimiterPos = str.find(delimiter)) != std::string::npos) {
        irPaths.push_back(str.substr(0, delimiterPos));
        str = str.substr(delimiterPos + 1);
    }
    irPaths.push_back(str);
    return irPaths;
}

} // namespace FuncTestUtils