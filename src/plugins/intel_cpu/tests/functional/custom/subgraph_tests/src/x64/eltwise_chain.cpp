// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/eltwise_chain.hpp"

#include <vector>

using namespace CPUTestUtils;

namespace ov {
namespace test {
using namespace ov::test::utils;
using namespace ov::test::eltwise_chain;

namespace {

// An i32 comparison is fused into a chain of i32 arithmetic operations.
// x64 only until the ARM64 and RISC-V i32 comparisons are exact (#39337).
std::vector<std::vector<EltwiseChainOp>> eltwiseOpsComparison() {
    std::vector<std::vector<EltwiseChainOp>> ops;
    for (const auto comparison : {ComparisonTypes::EQUAL,
                                  ComparisonTypes::NOT_EQUAL,
                                  ComparisonTypes::GREATER,
                                  ComparisonTypes::GREATER_EQUAL,
                                  ComparisonTypes::LESS,
                                  ComparisonTypes::LESS_EQUAL}) {
        ops.push_back({EltwiseTypes::ADD, EltwiseTypes::SUBTRACT, comparison});
    }
    return ops;
}

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChain_Comparison,
                         EltwiseChainTest,
                         ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapes())),
                                            ::testing::Values(InputLayerType::CONSTANT, InputLayerType::PARAMETER),
                                            ::testing::Values(std::vector<ElementType>(4, ElementType::i32)),
                                            ::testing::ValuesIn(eltwiseOpsComparison()),
                                            ::testing::Values(PostNode::EMPTY),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         EltwiseChainTest::getTestCaseName);

// An 8-bit SquaredDifference is computed in i32 by the snippets kernel.
// The parameters are in [10, 20), so the result never leaves the 8-bit range.
INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChain_SquaredDifference_8bit,
                         EltwiseChainTest,
                         ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapes())),
                                            ::testing::Values(InputLayerType::PARAMETER),
                                            ::testing::Values(std::vector<ElementType>(4, ElementType::i8),
                                                              std::vector<ElementType>(4, ElementType::u8)),
                                            ::testing::Values(std::vector<EltwiseChainOp>{EltwiseTypes::SQUARED_DIFF}),
                                            ::testing::Values(PostNode::EMPTY),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         EltwiseChainTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
