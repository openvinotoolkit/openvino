// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/prior_box.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    TEST_P(PriorBoxLayerTest, Serialize) {
        Serialize();
    }

    const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::U16
    };
    const std::vector<std::vector<float>> min_sizes = {
        {16.f, 32.f}
    };

    const std::vector<std::vector<float>> max_sizes = {
        {256.f, 512.f}
    };

    const std::vector<std::vector<float>> aspect_ratios = {
        {0.66f, 1.56f}
    };

    const std::vector<std::vector<float>> densities = {
        {0.55f}
    };

    const std::vector<std::vector<float>> fixed_ratios = {
        {0.88f}
    };

    const std::vector<std::vector<float>> fixed_sizes = {
        {1.25f}
    };

    const std::vector<bool> clips = {
        true, false
    };

    const std::vector<bool> flips = {
        true, false
    };

    const std::vector<float> steps = {
        1.0f, 2.0f
    };

    const std::vector<float> offsets = {
        0.0f, 0.5f
    };

    const std::vector<std::vector<float>> variances = {
        {2.22f, 3.14f}
    };

    const std::vector<bool> scale_all_sizes = {
        true, false
    };

    const std::vector<bool> min_max_aspect_ratios_order = {
        true, false
    };

    const std::vector<size_t> inputShape = {128, 128};
    const std::vector<size_t> imageShape = {50, 50};

    const auto layerSpecificParams = ::testing::Combine(
            ::testing::ValuesIn(min_sizes),
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
            ::testing::ValuesIn(scale_all_sizes),
            ::testing::ValuesIn(min_max_aspect_ratios_order));

    INSTANTIATE_TEST_SUITE_P(smoke_PriorBox_Basic, PriorBoxLayerTest,
                            ::testing::Combine(
                                layerSpecificParams,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(inputShape),
                                ::testing::Values(imageShape),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            PriorBoxLayerTest::getTestCaseName);
} // namespace
