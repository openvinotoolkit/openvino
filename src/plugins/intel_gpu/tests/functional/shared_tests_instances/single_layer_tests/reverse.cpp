// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/reverse.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16,
                                                               InferenceEngine::Precision::I32,
                                                               InferenceEngine::Precision::I64,
                                                               InferenceEngine::Precision::I8,
                                                               InferenceEngine::Precision::U8};

const std::vector<std::vector<size_t>> inputShapes1D = {{10}};
const std::vector<std::vector<int>> indices1D = {{0}};
const std::vector<std::string> modes = {"index", "mask"};

INSTANTIATE_TEST_SUITE_P(smoke_Reverse1D,
                         ReverseLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes1D),
                                            ::testing::ValuesIn(indices1D),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReverseLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> inputShapes2D = {{3, 4}, {1, 3}};
const std::vector<std::vector<int>> indices2D = {{0}, {1}};

INSTANTIATE_TEST_SUITE_P(smoke_Reverse2D,
                         ReverseLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes2D),
                                            ::testing::ValuesIn(indices2D),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReverseLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> inputShapes3D = {{1, 3, 4}, {2, 5, 6}};
const std::vector<std::vector<int>> indices3D = {{0}, {0, 1}, {0, 2}};
INSTANTIATE_TEST_SUITE_P(smoke_Reverse3D,
                         ReverseLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes3D),
                                            ::testing::ValuesIn(indices3D),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReverseLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> inputShapes4D = {{1, 2, 3, 4}, {1, 2, 5, 6}};
const std::vector<std::vector<int>> indices4D = {{1}, {1, 2}, {1, 3}};

INSTANTIATE_TEST_SUITE_P(smoke_Reverse4D,
                         ReverseLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes4D),
                                            ::testing::ValuesIn(indices4D),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReverseLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> inputShapes5D = {{1, 1, 4, 3, 3}};
const std::vector<std::vector<int>> indices5D = {{2}, {2, 3}, {2, 4}};

INSTANTIATE_TEST_SUITE_P(smoke_Reverse5D,
                         ReverseLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes5D),
                                            ::testing::ValuesIn(indices5D),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReverseLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> inputShapes6D = {{1, 1, 4, 3, 3, 3}};
const std::vector<std::vector<int>> indices6D = {{2}, {1, 3}, {3, 5}, {1, 4, 5}};

INSTANTIATE_TEST_SUITE_P(smoke_Reverse6D,
                         ReverseLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes6D),
                                            ::testing::ValuesIn(indices6D),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReverseLayerTest::getTestCaseName);

}  // namespace
