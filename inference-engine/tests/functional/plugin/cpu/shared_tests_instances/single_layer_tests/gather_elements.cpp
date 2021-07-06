// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/gather_elements.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> dPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I64,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I8
};
const std::vector<InferenceEngine::Precision> iPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I64
};

INSTANTIATE_TEST_SUITE_P(smoke_set1, GatherElementsLayerTest,
                        ::testing::Combine(
                            ::testing::Values(std::vector<size_t>({2, 2})),     // Data shape
                            ::testing::Values(std::vector<size_t>({2, 2})),     // Indices shape
                            ::testing::ValuesIn(std::vector<int>({-1, 0, 1})),  // Axis
                            ::testing::ValuesIn(dPrecisions),
                            ::testing::ValuesIn(iPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_set2, GatherElementsLayerTest,
                        ::testing::Combine(
                            ::testing::Values(std::vector<size_t>({2, 2, 1})),  // Data shape
                            ::testing::Values(std::vector<size_t>({4, 2, 1})),  // Indices shape
                            ::testing::ValuesIn(std::vector<int>({0, -3})),     // Axis
                            ::testing::ValuesIn(dPrecisions),
                            ::testing::ValuesIn(iPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_set3, GatherElementsLayerTest,
                        ::testing::Combine(
                            ::testing::Values(std::vector<size_t>({2, 2, 3, 5})),   // Data shape
                            ::testing::Values(std::vector<size_t>({2, 2, 3, 7})),   // Indices shape
                            ::testing::Values(3, -1),                               // Axis
                            ::testing::ValuesIn(dPrecisions),
                            ::testing::ValuesIn(iPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_set4, GatherElementsLayerTest,
                        ::testing::Combine(
                            ::testing::Values(std::vector<size_t>({3, 2, 3, 8})),   // Data shape
                            ::testing::Values(std::vector<size_t>({2, 2, 3, 8})),   // Indices shape
                            ::testing::Values(0, -4),                               // Axis
                            ::testing::ValuesIn(dPrecisions),
                            ::testing::ValuesIn(iPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_set5, GatherElementsLayerTest,
                        ::testing::Combine(
                            ::testing::Values(std::vector<size_t>({3, 2, 3, 4, 8})),   // Data shape
                            ::testing::Values(std::vector<size_t>({3, 2, 3, 5, 8})),   // Indices shape
                            ::testing::Values(3, -2),                                  // Axis
                            ::testing::ValuesIn(dPrecisions),
                            ::testing::ValuesIn(iPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherElementsLayerTest::getTestCaseName);
}  // namespace
