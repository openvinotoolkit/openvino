// Copyright (C) 2018-2021 Intel Corporation
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
std::vector<bool> mergeRepeated{true, false};

const auto basicCases = ::testing::Combine(
    ::testing::ValuesIn(netPrecisions),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::Values(std::vector<size_t>({ 50, 3, 3 }),
                      std::vector<size_t>({ 50, 3, 7 }),
                      std::vector<size_t>({ 50, 3, 8 }),
                      std::vector<size_t>({ 50, 3, 16 }),
                      std::vector<size_t>({ 50, 3, 128 }),
                      std::vector<size_t>({ 50, 3, 49 }),
                      std::vector<size_t>({ 50, 3, 55 }),
                      std::vector<size_t>({ 1, 1, 16 })),
    ::testing::ValuesIn(mergeRepeated),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_CtcGreedyDecoderBasic, CTCGreedyDecoderLayerTest,
                        basicCases,
                        CTCGreedyDecoderLayerTest::getTestCaseName);
}  // namespace
