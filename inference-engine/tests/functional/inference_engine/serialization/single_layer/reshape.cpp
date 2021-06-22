// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/reshape.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    TEST_P(ReshapeLayerTestRevise, Serialize) {
        Serialize();
    }

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };

    INSTANTIATE_TEST_SUITE_P(smoke_ReshapeSerialization, ReshapeLayerTestRevise,
            ::testing::Combine(
                ::testing::Values(true),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),
                ::testing::Values(std::vector<int64_t>({30, 30, 30, 30})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(std::map<std::string, std::string>({{CONFIG_KEY(DYN_BATCH_ENABLED), CONFIG_VALUE(YES)}}))),
                ReshapeLayerTestRevise::getTestCaseName);
}  // namespace
