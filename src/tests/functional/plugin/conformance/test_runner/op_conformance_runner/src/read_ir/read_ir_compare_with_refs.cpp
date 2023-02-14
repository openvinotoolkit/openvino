// Copyright (C) 2018-2023 Intel Corporation
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

#define _OPENVINO_OP_REG(NAME, NAMESPACE)                                                                  \
    INSTANTIATE_TEST_SUITE_P(conformance_##NAME,                                                           \
                             ReadIRTest,                                                                   \
                             ::testing::Combine(::testing::ValuesIn(getModelPaths(IRFolderPaths, #NAME)),  \
                                                ::testing::Values(targetDevice),                           \
                                                ::testing::Values(pluginConfig)),                          \
                             ReadIRTest::getTestCaseName); \

// It should point on latest opset which contains biggest list of operations
#include "openvino/opsets/opset10_tbl.hpp"
#undef _OPENVINO_OP_REG

INSTANTIATE_TEST_SUITE_P(conformance_other,
                        ReadIRTest,
                        ::testing::Combine(::testing::ValuesIn(getModelPaths(IRFolderPaths)),
                                        ::testing::Values(targetDevice),
                                        ::testing::Values(pluginConfig)),
                        ReadIRTest::getTestCaseName);

}  // namespace

}  // namespace op
}  // namespace conformance
}  // namespace test
}  // namespace ov
