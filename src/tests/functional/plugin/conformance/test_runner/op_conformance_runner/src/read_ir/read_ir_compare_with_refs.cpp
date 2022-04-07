// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"

#include "read_ir_test/read_ir_compare_with_refs.hpp"
#include "conformance.hpp"

namespace ov {
namespace test {
namespace conformance {
namespace op {

using namespace ov::test::subgraph;

namespace {



INSTANTIATE_TEST_SUITE_P(conformance,
                         ReadIRTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(getModelPaths(IRFolderPaths)),
                                 ::testing::Values(targetDevice),
                                 ::testing::Values(pluginConfig)),
                         ReadIRTest::getTestCaseName);
}  // namespace

}  // namespace op
}  // namespace conformance
}  // namespace test
}  // namespace ov
