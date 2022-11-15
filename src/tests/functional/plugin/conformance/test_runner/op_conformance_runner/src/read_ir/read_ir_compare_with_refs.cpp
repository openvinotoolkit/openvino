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

#include <ngraph/ngraph.hpp>

#define NGRAPH_OP(NAME, NAMESPACE)                                                                         \
    INSTANTIATE_TEST_SUITE_P(conformance##NAME,                                                            \
                             ReadIRTest,                                                                   \
                             ::testing::Combine(::testing::ValuesIn(getModelPaths(IRFolderPaths, #NAME)),  \
                                                ::testing::Values(targetDevice),                           \
                                                ::testing::Values(pluginConfig)),                          \
                             ReadIRTest::getTestCaseName); \

// It should point on latest opset which contains biggest list of operations
#include <ngraph/opsets/opset10_tbl.hpp>
#undef NGRAPH_OP

INSTANTIATE_TEST_SUITE_P(conformanceOther,
                        ReadIRTest,
                        ::testing::Combine(::testing::ValuesIn(getModelPaths(IRFolderPaths, CONFORMANCE_OTHER_OPS)),
                                        ::testing::Values(targetDevice),
                                        ::testing::Values(pluginConfig)),
                        ReadIRTest::getTestCaseName);

}  // namespace

}  // namespace op
}  // namespace conformance
}  // namespace test
}  // namespace ov
