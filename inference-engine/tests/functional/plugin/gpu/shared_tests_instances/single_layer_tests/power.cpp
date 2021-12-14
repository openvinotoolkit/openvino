// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/power.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

    std::vector<std::vector<std::vector<size_t>>> inShapes = {
            {{1, 8}},
            {{2, 16}},
            {{3, 32}},
            {{4, 64}},
            {{5, 128}},
            {{6, 256}},
            {{7, 512}},
            {{8, 1024}}
    };

    std::vector<std::vector<float >> Power = {
            {0.0f},
            {0.5f},
            {1.0f},
            {1.1f},
            {1.5f},
            {2.0f},
    };

    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                             InferenceEngine::Precision::FP16,
    };

    INSTANTIATE_TEST_SUITE_P(smoke_power, PowerLayerTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(inShapes),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                    ::testing::ValuesIn(Power)),
                            PowerLayerTest::getTestCaseName);
}  // namespace
