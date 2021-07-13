// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/grn.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;

namespace {
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
     // input shapes
    ::testing::Values(std::vector<size_t>{16, 24},
                      std::vector<size_t>{3, 16, 24},
                      std::vector<size_t>{1, 3, 30, 30},
                      std::vector<size_t>{2, 16, 15, 20}),
    // bias
    ::testing::Values(0.33f, 1.1f),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_CASE_P(smoke_Grn_Basic, GrnLayerTest,
                        basicCases,
                        GrnLayerTest::getTestCaseName);
}  // namespace
