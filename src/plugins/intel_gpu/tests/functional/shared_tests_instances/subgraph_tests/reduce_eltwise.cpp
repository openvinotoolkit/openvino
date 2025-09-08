// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/reduce_eltwise.hpp"

namespace {
using ov::test::ReduceEltwiseTest;

INSTANTIATE_TEST_SUITE_P(smoke_ReduceEltwise6D, ReduceEltwiseTest,
                        testing::Combine(
                                testing::Values(ov::Shape{2, 3, 4, 5, 6, 7}),
                                testing::Values(std::vector<int>{2, 3, 4}),
                                testing::Values(ov::test::utils::OpType::VECTOR),
                                testing::Values(false),
                                testing::Values(ov::element::f32),
                                testing::Values(ov::test::utils::DEVICE_GPU)),
                        ReduceEltwiseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReduceEltwise5D, ReduceEltwiseTest,
                        testing::Combine(
                                testing::Values(ov::Shape{2, 3, 4, 5, 6}),
                                testing::Values(std::vector<int>{2, 3}),
                                testing::Values(ov::test::utils::OpType::VECTOR),
                                testing::Values(false),
                                testing::Values(ov::element::f32),
                                testing::Values(ov::test::utils::DEVICE_GPU)),
                        ReduceEltwiseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReduceEltwise4D, ReduceEltwiseTest,
                        testing::Combine(
                                testing::Values(ov::Shape{2, 3, 4, 5}),
                                testing::Values(std::vector<int>{2}),
                                testing::Values(ov::test::utils::OpType::VECTOR),
                                testing::Values(false),
                                testing::Values(ov::element::f32),
                                testing::Values(ov::test::utils::DEVICE_GPU)),
                        ReduceEltwiseTest::getTestCaseName);

}  // namespace
