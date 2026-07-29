// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "conformance.hpp"

#include "functional_test_utils/skip_tests_config.hpp"

namespace ov {
namespace test {
namespace conformance {

const char *targetDevice = "";
const char *targetPluginName = "";
const char *refCachePath = "";

std::vector<std::string> IRFolderPaths = {};

ShapeMode shapeMode = ov::test::conformance::ShapeMode::BOTH;

} // namespace conformance
} // namespace test
} // namespace ov

const std::vector<std::regex>& disabled_test_patterns() {
    const static std::vector<std::regex> patterns{
        std::regex(R"(.*OVCompiledModelBaseTest.*import_from_.*_blob.*targetDevice=(MULTI|AUTO|CPU).*)"),
        std::regex(R"(.*OVCompiledModelBaseTest.*compile_from_.*_blob.*targetDevice=(MULTI|AUTO|CPU).*)"),
        std::regex(R"(.*OVCompiledModelBaseTest.*compile_from_cached_weightless_blob.*targetDevice=(MULTI|AUTO|CPU).*)"),
        std::regex(R"(.*OVCompiledModelBaseTest.*use_blob_hint_.*targetDevice=CPU.*)"),
    };

    return patterns;
}
