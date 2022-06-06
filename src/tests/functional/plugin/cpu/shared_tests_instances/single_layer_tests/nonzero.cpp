// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/nonzero.hpp"
#include "common_test_utils/test_constants.hpp"
#include <vector>

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {
    std::vector<ov::test::InputShape> inShapes = {
        {{}, {{1000}}},
        {{}, {{4, 1000}}},
        {{}, {{2, 4, 1000}}},
        {{}, {{2, 4, 4, 1000}}},
        {{}, {{2, 4, 4, 2, 1000}}},
    };

    std::vector<ov::test::InputShape> inShapesDynamic = {
        {
            // dynamic
            {-1},
            // target
            {
                {2},
                {6},
                {1},
                {2}
            }
        },
        {
            // dynamic
            {{1, 10}},
            // target
            {
                {2},
                {6},
                {4},
                {6}
            }
        },
        {
            // dynamic
            {-1, -1},
            // target
            {
                {4, 24},
                {8, 32},
                {4, 24},
                {16, 48}
            }
        },
        {
            // dynamic
            {{1, 50}, -1},
            // target
            {
                {1, 16},
                {8, 32},
                {4, 16},
                {8, 32}
            }
        },
        {
            // dynamic
            {-1, -1, -1},
            // target
            {
                {4, 24, 6},
                {8, 32, 10},
                {4, 24, 6},
                {16, 48, 12}
            }
        },
        {
            // dynamic
            {-1, {1, 50}, {1, 40}},
            // target
            {
                {1, 16, 8},
                {8, 32, 14},
                {4, 16, 10},
                {8, 32, 14}
            }
        },
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {4, 24, 6, 4},
                {8, 32, 10, 2},
                {4, 24, 6, 4},
                {16, 48, 12, 2}
            }
        },
        {
            // dynamic
            {-1, {1, 50}, {1, 20}, -1},
            // target
            {
                {1, 16, 8, 8},
                {8, 32, 14, 6},
                {4, 16, 10, 8},
                {8, 32, 14, 6},
            }
        },
            {
            // dynamic
            {-1, -1, -1, -1, -1},
            // target
            {
                {4, 24, 5, 6, 4},
                {8, 32, 9, 10, 2},
                {4, 24, 5, 6, 4},
                {16, 48, 9, 12, 2}
            }
        },
        {
            // dynamic
            {-1, {1, 50}, -1, {1, 30}, -1},
            // target
            {
                {1, 16, 1, 8, 8},
                {8, 32, 5, 14, 6},
                {4, 16, 9, 10, 8},
                {8, 32, 5, 14, 6},
            }
        }
    };

    const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::U8,
    };

    ConfigMap config;

    INSTANTIATE_TEST_SUITE_P(smoke_nonzeroStatic, NonZeroLayerTest,
                            ::testing::Combine(
                                ::testing::ValuesIn(inShapes),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(config)),
                            NonZeroLayerTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_nonzeroDynamic, NonZeroLayerTest,
                            ::testing::Combine(
                                ::testing::ValuesIn(inShapesDynamic),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(config)),
                            NonZeroLayerTest::getTestCaseName);
    } // namespace
