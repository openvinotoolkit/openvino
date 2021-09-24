// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/conversion.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::helpers::ConversionTypes> conversionOpTypes = {
    ngraph::helpers::ConversionTypes::CONVERT,
    ngraph::helpers::ConversionTypes::CONVERT_LIKE,
};

const std::vector<std::vector<size_t>> inShape = {{1, 2, 3, 4}};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    // Ticket: 59594
    // InferenceEngine::Precision::I4,
    InferenceEngine::Precision::I8,
    InferenceEngine::Precision::I16,
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::I64,
    // Ticket: 59594
    // InferenceEngine::Precision::BIN,
    // InferenceEngine::Precision::BOOL,
    // InferenceEngine::Precision::U4,
    InferenceEngine::Precision::U8,
    InferenceEngine::Precision::U16,
    // Ticket: 59594
    // InferenceEngine::Precision::U32,
    InferenceEngine::Precision::U64,
    InferenceEngine::Precision::BF16,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32};

INSTANTIATE_TEST_SUITE_P(smoke_ConversionLayerTest,
                         ConversionLayerTest,
                         ::testing::Combine(::testing::ValuesIn(conversionOpTypes),
                                            ::testing::Values(inShape),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         ConversionLayerTest::getTestCaseName);
}  // namespace
