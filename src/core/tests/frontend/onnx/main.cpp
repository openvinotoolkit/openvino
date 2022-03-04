// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <string>

#include "gtest/gtest.h"
#include "utils.hpp"

std::string get_disabled_tests() {
    std::string result = "-";
    const std::string manifest_path = MANIFEST;
    std::ifstream manifest_stream(manifest_path);
    std::string line;
    while (std::getline(manifest_stream, line)) {
        if (line.empty()) {
            continue;
        }
        if (line.size() > 0 && line[0] == '#') {
            continue;
        }
        result += ":" + line;
    }
    manifest_stream.close();
    return result;
}

int main(int argc, char** argv) {
    ::testing::GTEST_FLAG(filter) += get_disabled_tests();
    return FrontEndTestUtils::run_tests(argc, argv);
}
