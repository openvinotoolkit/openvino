// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"

#include "read_ir/read_ir_query_network.hpp"
#include "conformance.hpp"

namespace ConformanceTests {
using namespace LayerTestsDefinitions;

namespace {
INSTANTIATE_TEST_SUITE_P(conformance,
                        ReadIRTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(CommonTestUtils::getFileListByPatternRecursive(IRFolderPaths,  std::regex(R"(.*\.xml)"))),
                                ::testing::Values(targetDevice),
                                ::testing::Values(pluginConfig)),
                        ReadIRTest::getTestCaseName);
} // namespace
} // namespace ConformanceTests
