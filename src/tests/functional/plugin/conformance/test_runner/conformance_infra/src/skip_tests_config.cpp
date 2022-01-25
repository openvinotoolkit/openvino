// Copyright (C) 2018-2022 Intel Corporation
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

std::vector<std::string> IRFolderPaths = {};
std::vector<std::string> disabledTests = {};

ov::AnyMap pluginConfig = {};

} // namespace conformance
} // namespace test
} // namespace ov

std::vector<std::string> disabledTestPatterns() {
    return ov::test::conformance::disabledTests;
}
