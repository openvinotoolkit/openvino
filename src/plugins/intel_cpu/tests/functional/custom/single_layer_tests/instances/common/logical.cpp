// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/logical.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace logical {

const auto unaryEltwiseCases = ::testing::Combine(
    ::testing::ValuesIn(inUnaryShapes()),
    ::testing::ValuesIn(logicalUnaryTypes()),
    ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),  // skipped
    ::testing::ValuesIn(inferPrc()),
    ::testing::Values(false)
);
INSTANTIATE_TEST_SUITE_P(smoke_Logical_Unary_Eltwise_CPU, LogicalLayerCPUTest, unaryEltwiseCases, LogicalLayerCPUTest::getTestCaseName);

const auto unarySnippetsCases = ::testing::Combine(
    ::testing::ValuesIn(inUnaryShapes()),
    ::testing::ValuesIn(logicalUnaryTypesSnippets()),
    ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),  // skipped
    ::testing::ValuesIn(inferPrc()),
    ::testing::Values(true)
);
INSTANTIATE_TEST_SUITE_P(smoke_Logical_Unary_Snippets_CPU, LogicalLayerCPUTest, unarySnippetsCases, LogicalLayerCPUTest::getTestCaseName);

const auto binaryEltwiseCases = ::testing::Combine(
    ::testing::ValuesIn(inBinaryShapes()),
    ::testing::ValuesIn(logicalBinaryTypes()),
    ::testing::ValuesIn(secondInTypes()),
    ::testing::ValuesIn(inferPrc()),
    ::testing::Values(false)
);
INSTANTIATE_TEST_SUITE_P(smoke_Logical_Binary_Eltwise_CPU, LogicalLayerCPUTest, binaryEltwiseCases, LogicalLayerCPUTest::getTestCaseName);

const auto binarySnippetsCases = ::testing::Combine(
    ::testing::ValuesIn(inBinaryShapes()),
    ::testing::ValuesIn(logicalBinaryTypesSnippets()),
    ::testing::ValuesIn(secondInTypes()),
    ::testing::ValuesIn(inferPrc()),
    ::testing::Values(true)
);
INSTANTIATE_TEST_SUITE_P(smoke_Logical_Binary_Snippets_CPU, LogicalLayerCPUTest, binarySnippetsCases, LogicalLayerCPUTest::getTestCaseName);

}  // namespace logical
}  // namespace test
}  // namespace ov
