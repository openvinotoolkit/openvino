// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/conversion.hpp"

#include <vector>

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::helpers::ConversionTypes> conversionOpTypes = {
    ngraph::helpers::ConversionTypes::CONVERT,
    ngraph::helpers::ConversionTypes::CONVERT_LIKE,
};

const std::vector<std::vector<size_t>> inShape = {{1, 2, 3, 4}};

const std::vector<InferenceEngine::Precision> precisions = {
    InferenceEngine::Precision::BOOL, InferenceEngine::Precision::BIN,
    InferenceEngine::Precision::U4,   InferenceEngine::Precision::U8,
    InferenceEngine::Precision::I4,   InferenceEngine::Precision::I8,
    InferenceEngine::Precision::U16,  InferenceEngine::Precision::I16,
    InferenceEngine::Precision::U32,  InferenceEngine::Precision::I32,
    InferenceEngine::Precision::U64,  InferenceEngine::Precision::I64,
    InferenceEngine::Precision::BF16, InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP64};

TEST_P(ConversionLayerTest, Serialize) {
    Serialize();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Serialization_ConversionLayerTest, ConversionLayerTest,
    ::testing::Combine(::testing::ValuesIn(conversionOpTypes),
                       ::testing::Values(inShape),
                       ::testing::ValuesIn(precisions),
                       ::testing::ValuesIn(precisions),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(InferenceEngine::Layout::ANY),
                       ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    ConversionLayerTest::getTestCaseName);

}  // namespace
