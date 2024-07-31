// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/integer_reduce_mean.hpp"

#include <tuple>
#include <vector>

using namespace ov::test;
namespace {

const std::vector<ov::element::Type> input_precision = {ov::element::f32};
const std::vector<ov::element::Type> integer_input_precision = {ov::element::i32, ov::element::i8, ov::element::u8};
const std::vector<std::vector<size_t>> input_shape = {{1, 2, 3, 3}};
const std::vector<std::vector<size_t>> axes = {{2, 3}};

INSTANTIATE_TEST_SUITE_P(smoke_ReduceMeanQuantized,
                         IntegerReduceMeanTest,
                         testing::Combine(
                            ::testing::ValuesIn(input_precision),
                            ::testing::ValuesIn(input_shape),
                            ::testing::ValuesIn(axes),
                            ::testing::Values(true),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         IntegerReduceMeanTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReduceMeanIntegerInput,
                         IntegerReduceMeanTest,
                         testing::Combine(
                            ::testing::ValuesIn(integer_input_precision),
                            ::testing::ValuesIn(input_shape),
                            ::testing::ValuesIn(axes),
                            ::testing::Values(false),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         IntegerReduceMeanTest::getTestCaseName);

}  // namespace
