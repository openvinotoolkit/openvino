// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/eltwise_overflow.hpp"

namespace ov {
namespace test {

const std::vector<EltwiseOverflowKind> overflowKinds = {EltwiseOverflowKind::UNDERFLOW, EltwiseOverflowKind::OVERFLOW};

const std::vector<ov::Shape> testShapes = {
    {4},           // small 1D
    {64},          // larger 1D to exercise vectorized JIT path
    {1, 2, 2, 2},  // 4D typical NN shape
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseOverflowU8,
                         EltwiseOverflowLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(overflowKinds), ::testing::ValuesIn(testShapes)),
                         EltwiseOverflowLayerCPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov
