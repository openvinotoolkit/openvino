// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/bucketize.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<std::vector<size_t>> dataShapes = {
    {1, 20, 20},
    {2, 3, 50, 50}
};

const std::vector<std::vector<size_t>> bucketsShapes = {
    {5},
    {20},
    {100}
};

const std::vector<InferenceEngine::Precision> inPrc = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I64,
    InferenceEngine::Precision::I32
};

const std::vector<InferenceEngine::Precision> netPrc = {
    InferenceEngine::Precision::I64,
    InferenceEngine::Precision::I32
};

const auto test_Bucketize_right_edge = ::testing::Combine(
    ::testing::ValuesIn(dataShapes),
    ::testing::ValuesIn(bucketsShapes),
    ::testing::Values(true),
    ::testing::ValuesIn(inPrc),
    ::testing::ValuesIn(inPrc),
    ::testing::ValuesIn(netPrc),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto test_Bucketize_left_edge = ::testing::Combine(
    ::testing::ValuesIn(dataShapes),
    ::testing::ValuesIn(bucketsShapes),
    ::testing::Values(false),
    ::testing::ValuesIn(inPrc),
    ::testing::ValuesIn(inPrc),
    ::testing::ValuesIn(netPrc),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsBucketize_right, BucketizeLayerTest, test_Bucketize_right_edge, BucketizeLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsBucketize_left, BucketizeLayerTest, test_Bucketize_left_edge, BucketizeLayerTest::getTestCaseName);
