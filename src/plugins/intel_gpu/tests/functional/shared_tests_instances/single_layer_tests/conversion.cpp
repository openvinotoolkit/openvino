// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/conversion.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::helpers::ConversionTypes> conversionOpTypes = {
    ngraph::helpers::ConversionTypes::CONVERT,
    ngraph::helpers::ConversionTypes::CONVERT_LIKE,
};

const std::vector<std::vector<size_t>> inShape = {{1, 2, 3, 4}};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I8,
};

INSTANTIATE_TEST_SUITE_P(smoke_NoReshape, ConversionLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(conversionOpTypes),
                                ::testing::Values(inShape),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConversionLayerTest::getTestCaseName);

}  // namespace
