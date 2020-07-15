// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fake_quantize_precision_selection_transformation.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ngraph_functions/low_precision_transformations/fake_quantize_function.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsFactory::createParamsU8I8()
};

const std::vector<LayerTestsUtils::LayerTransformation::LptVersion> versionValues = {
    LayerTestsUtils::LayerTransformation::LptVersion::cnnNetwork,
    LayerTestsUtils::LayerTransformation::LptVersion::nGraph
};

const std::vector<FakeQuantizePrecisionSelectionTransformationTestValues> testValues = {
    {
        { ngraph::element::u8, ngraph::element::i8 },
        { ngraph::element::u8 },
        true,
        {
            { 256ul, { }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
            { 255ul, { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -12.7f }, { 12.7f } }
        },
        {
            ngraph::element::u8,
            { 256ul, { }, { 0.f }, { 2.55f }, { 0.f }, { 255.f } },
            { }
        },
    },
    {
        { ngraph::element::u8, ngraph::element::i8 },
        { ngraph::element::i8 },
        // INT8 is not available for limited operation (Convolution)
        false,
        {
            { 256ul, { }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
            { 255ul, { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -12.7f }, { 12.7f } }
        },
        {
            // original precision is used
            ngraph::element::u8,
            // FakeQuantize has to select the first available: U8, not limited operation required I8 but this fact doesn't affect
            { 256ul, { }, { 0.f }, { 25.5f }, { 0.f }, { 255.f } },
            // FakeQuantize on weights is not changed
            { 255ul, { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -12.7f }, { 12.7f } }
        },
    },
};

// TODO: add something to avoid cleanup and enable
INSTANTIATE_TEST_CASE_P(LPT, FakeQuantizePrecisionSelectionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::SizeVector({ 1, 32, 72, 48 })),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(versionValues),
        ::testing::ValuesIn(testValues)),
    FakeQuantizePrecisionSelectionTransformation::getTestCaseName);
}  // namespace
