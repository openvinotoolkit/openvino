// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/concat.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
// TODO: All concat on axis 0 always fails by accuracy
std::vector<size_t > axes = {1, 2, 3};
std::vector<std::vector<std::vector<size_t>>> inShapes = {
        {{10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}}
};
std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};


INSTANTIATE_TEST_CASE_P(Axis_1_and_3, ConcatLayerTest,
                        ::testing::Combine(
                                ::testing::Values(1, 3),
                                ::testing::ValuesIn(inShapes),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        ConcatLayerTest::getTestCaseName);


// TODO: concat on axis 2 fails by accuracy with input precision different from FP16
INSTANTIATE_TEST_CASE_P(Axis_2, ConcatLayerTest,
                        ::testing::Combine(
                                ::testing::Values(2),
                                ::testing::ValuesIn(inShapes),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        ConcatLayerTest::getTestCaseName);
}  // namespace