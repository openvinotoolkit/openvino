// Copyright (C) 2018-2025 Intel Corporation
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
std::vector<std::string> disabledTests = {
    R"(.*OVCompiledModelBaseTest.*import_from_.*_blob.*targetDevice=(MULTI|AUTO|CPU).*)",
    R"(.*OVCompiledModelBaseTest.*compile_from_.*_blob.*targetDevice=(MULTI|AUTO|CPU).*)",
    R"(.*OVCompiledModelBaseTest.*compile_from_cached_weightless_blob.*targetDevice=(MULTI|AUTO|CPU).*)",
    R"(.*OVCompiledModelBaseTest.*use_blob_hint_.*targetDevice=CPU.*)",
};

ShapeMode shapeMode = ov::test::conformance::ShapeMode::BOTH;

} // namespace conformance
} // namespace test
} // namespace ov

std::vector<std::string> disabledTestPatterns() {
    return ov::test::conformance::disabledTests;
}
