// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/split.hpp"

using namespace LayerTestsDefinitions;

namespace {

TEST_P(SplitLayerTest, Serialize) {
    Serialize();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::U16,
        InferenceEngine::Precision::BOOL};

INSTANTIATE_TEST_SUITE_P(
    smoke_Split_Serialization, SplitLayerTest,
    ::testing::Combine(
        ::testing::Values(1, 2, 5, 10),
        ::testing::Values(0, 1, 2, 3),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{20, 30, 50, 50}),
        ::testing::Values(std::vector<size_t>({})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    SplitLayerTest::getTestCaseName
);

}   // namespace
