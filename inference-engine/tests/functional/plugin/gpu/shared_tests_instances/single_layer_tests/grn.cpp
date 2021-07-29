// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/grn.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;

namespace {
    // Common params
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };

    const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({ 1, 3, 30, 30 }),
                            std::vector<size_t>({ 2, 16, 15, 20})),
        ::testing::Values(0.33f, 1.1f),
        ::testing::Values(CommonTestUtils::DEVICE_GPU));

    INSTANTIATE_TEST_SUITE_P(smoke_Grn_Basic, GrnLayerTest,
                            basicCases,
                            GrnLayerTest::getTestCaseName);

}  // namespace
