// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/prior_box_clustered.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    TEST_P(PriorBoxClusteredLayerTest, Serialize) {
        serialize();
    }

    const std::vector<ov::test::ElementType> netPrecisions = {
        ov::test::ElementType::f32,
        ov::test::ElementType::f16
    };

    const std::vector<std::vector<float>> widths = {
        { 5.12f, 14.6f, 13.5f },
        { 7.0f, 8.2f, 33.39f }
    };

    const std::vector<std::vector<float>> heights = {
        { 15.12f, 15.6f, 23.5f },
        { 10.0f, 16.2f, 36.2f }
    };

    const std::vector<float> step_widths = {
        0.0f, 2.0f
    };

    const std::vector<float> step_heights = {
        0.0f, 1.5f
    };

    const std::vector<float> step = {
        0.0f
    };

    const std::vector<float> offsets = {
        0.5f
    };

    const std::vector<std::vector<float>> variances = {
        { 0.1f, 0.1f, 0.2f, 0.2f }
    };

    const std::vector<bool> clips = {
        true, false
    };

    const std::vector<ov::test::InputShape> inputShapes = {
        {{4, 4}, {{4, 4}}}
    };

    const std::vector<ov::test::InputShape> imageShapes = {
        {{50, 50}, {{50, 50}}}
    };

    const auto layerSpeficParams = ::testing::Combine(
        ::testing::ValuesIn(widths),
        ::testing::ValuesIn(heights),
        ::testing::ValuesIn(clips),
        ::testing::ValuesIn(step_widths),
        ::testing::ValuesIn(step_heights),
        ::testing::ValuesIn(step),
        ::testing::ValuesIn(offsets),
        ::testing::ValuesIn(variances));

    INSTANTIATE_TEST_SUITE_P(smoke_PriorBoxClustered_Basic, PriorBoxClusteredLayerTest,
            ::testing::Combine(
                layerSpeficParams,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(ov::test::ElementType::undefined),
                ::testing::Values(ov::test::ElementType::undefined),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::ValuesIn(inputShapes),
                ::testing::ValuesIn(imageShapes),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            PriorBoxClusteredLayerTest::getTestCaseName);

}  // namespace
