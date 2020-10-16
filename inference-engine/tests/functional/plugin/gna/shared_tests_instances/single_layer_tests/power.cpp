// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/power.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
    };

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
    };

    std::map<std::string, std::string> additional_config = {
            {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
            {"GNA_COMPACT_MODE", "NO"},
            {"GNA_PRECISION", "I16"},
    };

    INSTANTIATE_TEST_CASE_P(smoke_power, PowerLayerTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(inShapes),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                    ::testing::ValuesIn(Power),
                                    ::testing::Values(additional_config)),
                            PowerLayerTest::getTestCaseName);
}  // namespace
