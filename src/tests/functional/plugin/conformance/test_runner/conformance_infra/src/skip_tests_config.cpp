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
std::vector<std::string> disabledTests = {
    // GPU plugin does not support BF16
    R"(.*OVInferRequestCheckTensorPrecision.*get(Input|Output|Inputs|Outputs)From.*FunctionWith(Single|Several).*type=(bf16)_target_device=.*GPU.*)",
};

ov::AnyMap pluginConfig = {};

ShapeMode shapeMode = ov::test::conformance::ShapeMode::BOTH;

} // namespace conformance
} // namespace test
} // namespace ov

std::vector<std::string> disabledTestPatterns() {
    return ov::test::conformance::disabledTests;
}
