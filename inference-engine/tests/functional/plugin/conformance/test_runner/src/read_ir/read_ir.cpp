// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/path_utils.hpp"
#include "read_ir/read_ir.hpp"

namespace ConformanceTests {
using namespace LayerTestsDefinitions;

const char* targetDevice = "";
std::vector<std::string> IRFolderPaths = {};

namespace {
INSTANTIATE_TEST_CASE_P(conformance,
                        ReadIRTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(FuncTestUtils::getXmlPathsFromFolderRecursive(IRFolderPaths)),
                                ::testing::Values(targetDevice)),
                        ReadIRTest::getTestCaseName);
} // namespace
} // namespace ConformanceTests
