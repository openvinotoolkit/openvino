// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/grn.hpp"

using namespace LayerTestsDefinitions;

namespace {
    TEST_P(GrnLayerTest, Serialize) {
    Serialize();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

const auto basicCases = ::testing::Combine(
    ::testing::ValuesIn(netPrecisions),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::Values(InferenceEngine::Layout::ANY),
     // input shapes
    ::testing::Values(std::vector<size_t>{2, 16, 15, 20}),
    // bias
    ::testing::Values(1e-6f),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_CASE_P(smoke_GRN_Serialization, GrnLayerTest,
                        basicCases,
                        GrnLayerTest::getTestCaseName);
}  // namespace
