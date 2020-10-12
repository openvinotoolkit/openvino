// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/ctc_greedy_decoder.hpp"
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
        ::testing::Values(std::vector<size_t>({ 10, 1, 16 }),
                          std::vector<size_t>({ 20, 2, 8 })),
        ::testing::Values(true, false),
        ::testing::Values(CommonTestUtils::DEVICE_GPU));

    INSTANTIATE_TEST_CASE_P(smoke_CTC_Greedy_decoder_Basic, CTCGreedyDecoderLayerTest,
                            basicCases,
                            CTCGreedyDecoderLayerTest::getTestCaseName);
}  // namespace
