// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <fstream>

#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

namespace FuncTestUtils {
namespace SkipTestsConfig {

bool disable_tests_skipping = false;

bool currentTestIsDisabled() {
    bool skip_test = false;
    const auto fullName = ::testing::UnitTest::GetInstance()->current_test_info()->test_case_name()
                          + std::string(".") + ::testing::UnitTest::GetInstance()->current_test_info()->name();
    for (const auto &pattern : disabledTestPatterns()) {
        std::regex re(pattern);
        if (std::regex_match(fullName, re))
            skip_test = true;
    }
    return skip_test && !disable_tests_skipping;
}

std::vector<std::string> readSkipTestConfigFiles(const std::vector<std::string>& filePaths) {
    std::vector<std::string> res;
    for (const auto& filePath : filePaths) {
        if (!CommonTestUtils::fileExists(filePath)) {
            std::string msg = "Input directory (" + filePath + ") doesn't not exist!";
            throw std::runtime_error(msg);
        }
        std::ifstream file(filePath);
        if (file.is_open()) {
            std::string buffer;
            while (getline(file, buffer)) {
                if (buffer.find("#") == std::string::npos && !buffer.empty()) {
                    res.emplace_back(buffer);
                }
            }
        } else {
            std::string msg = "Error in opening file: " + filePath;
            throw std::runtime_error(msg);
        }
        file.close();
    }
    return res;
}

}  // namespace SkipTestsConfig
}  // namespace FuncTestUtils
