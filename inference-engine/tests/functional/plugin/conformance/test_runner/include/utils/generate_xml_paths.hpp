// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <regex>

#include "common_test_utils/file_utils.hpp"

namespace ConformanceTests {
namespace Utils {

static std::vector<std::string> generateXMLpaths(const std::vector<std::string> &folderPaths) {
    auto getIRnames = [](const std::string &folderPath) {
        std::vector<std::string> a;
        CommonTestUtils::directoryFileListRecursive(folderPath, a);
        std::set<std::string> res;
        for (auto &file : a) {
            if (CommonTestUtils::fileExists(file) && std::regex_match(file, std::regex(R"(.*\.xml)"))) {
                res.insert(file);
            }
        }
        return res;
    };

    std::vector<std::string> res;
    for (auto &&folderPath : folderPaths) {
        if (!CommonTestUtils::directoryExists(folderPath)) {
            continue;
        }
        auto IRs = getIRnames(folderPath);
        res.insert(res.end(), IRs.begin(), IRs.end());
    }
    return res;
}

} // namespace Utils
} // namespace ConformanceTests