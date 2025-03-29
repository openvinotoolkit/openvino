// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/cum_sum.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::CumSumLayerTest;

std::vector<std::vector<ov::Shape>> inShapes = {
        {{10, 10}},
        {{10, 10, 10}},
        {{10, 10, 10, 10}},
        {{10, 10, 10, 10, 10}},
        {{10, 10, 10, 10, 10, 10}},
};
std::vector<int64_t> axes = {-1, 0, 1};
std::vector<bool> exclusive = {false, true};
std::vector<bool> reverse = {false, true};
std::vector<ov::element::Type> precisions = {ov::element::f32,
                                             ov::element::f16};

std::vector<std::vector<ov::Shape>> shape1d = {{{10}}};
std::vector<int64_t> axis1d = {0};
INSTANTIATE_TEST_SUITE_P(smoke_CumSum1D, CumSumLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shape1d)),
                                ::testing::ValuesIn(precisions),
                                ::testing::ValuesIn(axis1d),
                                ::testing::ValuesIn(exclusive),
                                ::testing::ValuesIn(reverse),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        CumSumLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CumSum, CumSumLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                                ::testing::ValuesIn(precisions),
                                ::testing::ValuesIn(axes),
                                ::testing::ValuesIn(exclusive),
                                ::testing::ValuesIn(reverse),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        CumSumLayerTest::getTestCaseName);

const std::vector<std::vector<ov::Shape>> inShapesWithBigDims = {
        {{64, 64}},
        {{73, 73, 73}},
        {{49, 49, 49, 49}},
};

INSTANTIATE_TEST_SUITE_P(smoke_CumSumBigDims, CumSumLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesWithBigDims)),
                                ::testing::Values(ov::element::f32),
                                ::testing::ValuesIn(axes),
                                ::testing::ValuesIn(exclusive),
                                ::testing::ValuesIn(reverse),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        CumSumLayerTest::getTestCaseName);
}  // namespace
