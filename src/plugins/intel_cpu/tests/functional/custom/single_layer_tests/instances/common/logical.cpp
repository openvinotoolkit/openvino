// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/logical.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace logical {

const auto unaryCases = ::testing::Combine(
    ::testing::ValuesIn(inUnaryShapes()),
    ::testing::ValuesIn(logicalUnaryTypes()),
    ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),  // skipped
    ::testing::ValuesIn(enforceSnippets())
);
INSTANTIATE_TEST_SUITE_P(smoke_Logical_Unary, LogicalLayerCPUTest, unaryCases, LogicalLayerCPUTest::getTestCaseName);

const auto binaryCases = ::testing::Combine(
    ::testing::ValuesIn(inBinaryShapes()),
    ::testing::ValuesIn(logicalBinaryTypes()),
    ::testing::ValuesIn(secondInTypes()),
    ::testing::ValuesIn(enforceSnippets())
);
INSTANTIATE_TEST_SUITE_P(smoke_Logical_Binary, LogicalLayerCPUTest, binaryCases, LogicalLayerCPUTest::getTestCaseName);

}  // namespace logical
}  // namespace test
}  // namespace ov
