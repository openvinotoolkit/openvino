// copyright (c) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

std::string FrontEndTestUtils::get_disabled_tests(const std::string& manifest_path) {
    std::string result = ":-";
    std::ifstream manifest_stream(manifest_path);

    for (std::string line; std::getline(manifest_stream, line);) {
        if (line.size() && (line[0] != '#')) {
            result += ":" + line;
        }
    }
    return result;
}
