// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/cum_sum.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

std::vector<std::vector<size_t>> inShapes = {
        {10, 10},
        {10, 10, 10},
        {10, 10, 10, 10},
        {10, 10, 10, 10, 10},
        {10, 10, 10, 10, 10, 10},
};
std::vector<int64_t> axes = {-1, 0, 1};
std::vector<bool> exclusive = {false, true};
std::vector<bool> reverse = {false, true};
std::vector<InferenceEngine::Precision> precisions = {InferenceEngine::Precision::FP32,
                                                      InferenceEngine::Precision::FP16};

std::vector<std::vector<size_t>> shape1d = {{10}};
std::vector<int64_t> axis1d = {0};
INSTANTIATE_TEST_SUITE_P(smoke_CumSum1D, CumSumLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(shape1d),
                                ::testing::ValuesIn(precisions),
                                ::testing::ValuesIn(axis1d),
                                ::testing::ValuesIn(exclusive),
                                ::testing::ValuesIn(reverse),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        CumSumLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CumSum, CumSumLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes),
                                ::testing::ValuesIn(precisions),
                                ::testing::ValuesIn(axes),
                                ::testing::ValuesIn(exclusive),
                                ::testing::ValuesIn(reverse),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        CumSumLayerTest::getTestCaseName);
}  // namespace
