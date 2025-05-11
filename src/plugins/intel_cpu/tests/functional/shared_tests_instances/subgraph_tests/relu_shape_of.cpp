// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/relu_shape_of.hpp"

#include <vector>

using namespace ov::test;

namespace {
const std::vector<ov::element::Type> input_types = {ov::element::i32};

INSTANTIATE_TEST_SUITE_P(smoke_Check,
                         ReluShapeOfSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(input_types),
                                            ::testing::Values(ov::element::i64),
                                            ::testing::Values(ov::Shape{20, 10, 10, 10}),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ReluShapeOfSubgraphTest::getTestCaseName);
}  // namespace
