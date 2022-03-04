// Copyright (C) 2018-2022 Intel Corporation
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
    InferenceEngine::Precision::U8,
    InferenceEngine::Precision::I8,
    InferenceEngine::Precision::U16,
    InferenceEngine::Precision::I16,
    InferenceEngine::Precision::U32,
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::U64,
    InferenceEngine::Precision::I64,
    InferenceEngine::Precision::BF16,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP64,
    InferenceEngine::Precision::BOOL,
    InferenceEngine::Precision::MIXED,
    InferenceEngine::Precision::Q78,
    InferenceEngine::Precision::U4,
    InferenceEngine::Precision::I4,
    InferenceEngine::Precision::BIN,
    InferenceEngine::Precision::CUSTOM,
};

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
