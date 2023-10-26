// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/prior_box.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::I32,
                                                               InferenceEngine::Precision::U16};

const std::vector<std::vector<float>> min_sizes = {{256.0f}};

const std::vector<std::vector<float>> max_sizes = {{315.0f}};

const std::vector<std::vector<float>> aspect_ratios = {{2.0f}};

const std::vector<std::vector<float>> densities = {{1.0f}};

const std::vector<std::vector<float>> fixed_ratios = {{}};

const std::vector<std::vector<float>> fixed_sizes = {{}};

const std::vector<bool> clips = {false, true};

const std::vector<bool> flips = {false, true};

const std::vector<float> steps = {
    1.0f,
};

const std::vector<float> offsets = {
    0.0f,
};

const std::vector<std::vector<float>> variances = {{}};

const std::vector<bool> min_max_aspect_ratios_order = {false, true};

const std::vector<size_t> inputShape = {300, 300};
const std::vector<size_t> imageShape = {32, 32};

const auto scaleSizesParams = ::testing::Combine(::testing::ValuesIn(min_sizes),
                                                    ::testing::ValuesIn(max_sizes),
                                                    ::testing::ValuesIn(aspect_ratios),
                                                    ::testing::ValuesIn(densities),
                                                    ::testing::ValuesIn(fixed_ratios),
                                                    ::testing::ValuesIn(fixed_sizes),
                                                    ::testing::ValuesIn(clips),
                                                    ::testing::ValuesIn(flips),
                                                    ::testing::ValuesIn(steps),
                                                    ::testing::ValuesIn(offsets),
                                                    ::testing::ValuesIn(variances),
                                                    ::testing::Values(true),
                                                    ::testing::ValuesIn(min_max_aspect_ratios_order));

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_PriorBox8_Scale,
                         PriorBoxLayerTest,
                         ::testing::Combine(scaleSizesParams,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::I32),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(inputShape),
                                            ::testing::Values(imageShape),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         PriorBoxLayerTest::getTestCaseName);

const auto noScaleSizesParams = ::testing::Combine(::testing::ValuesIn(min_sizes),
                                                    ::testing::ValuesIn(max_sizes),
                                                    ::testing::ValuesIn(aspect_ratios),
                                                    ::testing::ValuesIn(densities),
                                                    ::testing::ValuesIn(fixed_ratios),
                                                    ::testing::ValuesIn(fixed_sizes),
                                                    ::testing::ValuesIn(clips),
                                                    ::testing::ValuesIn(flips),
                                                    ::testing::ValuesIn(steps),
                                                    ::testing::ValuesIn(offsets),
                                                    ::testing::ValuesIn(variances),
                                                    ::testing::Values(false),
                                                    ::testing::ValuesIn(min_max_aspect_ratios_order));

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_PriorBox8_NoScale,
                         PriorBoxLayerTest,
                         ::testing::Combine(scaleSizesParams,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::I32),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(inputShape),
                                            ::testing::Values(imageShape),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         PriorBoxLayerTest::getTestCaseName);
