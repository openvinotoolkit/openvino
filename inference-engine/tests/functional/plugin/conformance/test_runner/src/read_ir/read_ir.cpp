// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"

#include "read_ir/read_ir.hpp"

namespace ConformanceTests {
using namespace LayerTestsDefinitions;

const char* targetDevice = "";
std::vector<std::string> IRFolderPaths = {};

namespace {
INSTANTIATE_TEST_CASE_P(conformance,
                        ReadIRTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(CommonTestUtils::getFileListByPatternRecursive(IRFolderPaths,  std::regex(R"(.*\.xml)"))),
                                ::testing::Values(targetDevice)),
                        ReadIRTest::getTestCaseName);
} // namespace
} // namespace ConformanceTests
