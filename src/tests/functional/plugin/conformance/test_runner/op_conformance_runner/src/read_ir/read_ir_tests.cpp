// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"

#include "read_ir_test/read_ir.hpp"

#include "conformance.hpp"

namespace ov {
namespace test {
namespace conformance {
namespace op {

using namespace ov::test::subgraph;

namespace {

TEST_P(ReadIRTest, Inference) {
    run();
}

TEST_P(ReadIRTest, QueryModel) {
    query_model();
}

TEST_P(ReadIRTest, ImportExport) {
    import_export();
}

#define _OPENVINO_OP_REG(NAME, NAMESPACE)                                                                  \
    INSTANTIATE_TEST_SUITE_P(conformance_##NAME,                                                           \
                             ReadIRTest,                                                                   \
                             ::testing::Combine(::testing::ValuesIn(getModelPaths(IRFolderPaths, #NAME)),  \
                                                ::testing::Values(targetDevice),                           \
                                                ::testing::Values(pluginConfig)),                          \
                             ReadIRTest::getTestCaseName); \

// It should point on latest opset which contains biggest list of operations
#include "openvino/opsets/opset12_tbl.hpp"
#undef _OPENVINO_OP_REG

INSTANTIATE_TEST_SUITE_P(conformance_subgraph,
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
