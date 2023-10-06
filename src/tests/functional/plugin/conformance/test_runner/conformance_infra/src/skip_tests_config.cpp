// Copyright (C) 2018-2023 Intel Corporation
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
std::vector<std::string> disabledTests = {};

ov::AnyMap pluginConfig = {};

ShapeMode shapeMode = ov::test::conformance::ShapeMode::BOTH;

} // namespace conformance
} // namespace test
} // namespace ov

std::vector<std::string> disabledTestPatterns() {
    if (!ov::with_cpu_x86_avx512_core()) {
        // on platforms which do not support bfloat16, we are disabling bf16 tests since there are no bf16 primitives,
        // tests are useless on such platforms
        ov::test::conformance::disabledTests.emplace_back(R"(.*(BF|bf)16.*)");
        ov::test::conformance::disabledTests.emplace_back(R"(.*bfloat16.*)");
    }
    return ov::test::conformance::disabledTests;
}
