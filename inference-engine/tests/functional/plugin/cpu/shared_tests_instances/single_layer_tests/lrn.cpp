// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/lrn.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<InferenceEngine::Precision> netPrecisions{
    InferenceEngine::Precision::FP32
};
const double alpha = 9.9e-05;
const double beta = 2;
const double bias = 1.0;
const size_t size = 5;

namespace LRN2D {

const std::vector<std::vector<int64_t>> axes = {{1}};

INSTANTIATE_TEST_CASE_P(smoke_LrnCheck_2D, LrnLayerTest,
                        ::testing::Combine(::testing::Values(alpha),
                                           ::testing::Values(beta),
                                           ::testing::Values(bias),
                                           ::testing::Values(size),
                                           ::testing::ValuesIn(axes),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(std::vector<size_t>({10, 16})),
                                           ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        LrnLayerTest::getTestCaseName);

} // namespace LRN2D

namespace LRN3D {

const std::vector<std::vector<int64_t>> axes = {{1}, {2}};

INSTANTIATE_TEST_CASE_P(smoke_LrnCheck_3D, LrnLayerTest,
                        ::testing::Combine(::testing::Values(alpha),
                                           ::testing::Values(beta),
                                           ::testing::Values(bias),
                                           ::testing::Values(size),
                                           ::testing::ValuesIn(axes),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(std::vector<size_t>({6, 10, 16})),
                                           ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        LrnLayerTest::getTestCaseName);

} // namespace LRN3D

namespace LRN4D {

const std::vector<std::vector<int64_t>> axes = {{1}, {2, 3}, {3, 2}};

INSTANTIATE_TEST_CASE_P(smoke_LrnCheck_4D, LrnLayerTest,
                        ::testing::Combine(::testing::Values(alpha),
                                           ::testing::Values(beta),
                                           ::testing::Values(bias),
                                           ::testing::Values(size),
                                           ::testing::ValuesIn(axes),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(std::vector<size_t>({10, 10, 3, 8})),
                                           ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        LrnLayerTest::getTestCaseName);

} // namespace LRN4D

namespace LRN5D {

const std::vector<std::vector<int64_t>> axes = {{1}, {2, 3, 4}, {4, 2, 3}};

INSTANTIATE_TEST_CASE_P(smoke_LrnCheck_5D, LrnLayerTest,
                        ::testing::Combine(::testing::Values(alpha),
                                           ::testing::Values(beta),
                                           ::testing::Values(bias),
                                           ::testing::Values(size),
                                           ::testing::ValuesIn(axes),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(std::vector<size_t>({1, 10, 10, 7, 4})),
                                           ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        LrnLayerTest::getTestCaseName);

} // namespace LRN5D
