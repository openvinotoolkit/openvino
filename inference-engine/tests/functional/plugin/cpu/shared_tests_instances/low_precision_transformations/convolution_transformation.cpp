// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/convolution_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16
};

// TODO: not used
const std::vector<InferenceEngine::details::LayerTransformation::Params> trasformationParamValues = {
    // LayerTestsUtils::LayerTransformationParamsFactory::createParams().setUpdatePrecisions(true),
    // LayerTestsUtils::LayerTransformationParamsFactory::createParams().setUpdatePrecisions(false),
    LayerTestsUtils::LayerTransformationParamsFactory::createParamsU8I8()
};

const std::vector<LayerTestsUtils::LayerTransformation::LptVersion> versions = {
    LayerTestsUtils::LayerTransformation::LptVersion::cnnNetwork,
    LayerTestsUtils::LayerTransformation::LptVersion::nGraph
};

const std::vector<LayerTestsDefinitions::ConvolutionTransformationParam> params = {
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        { 255ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
    },
    // {
    //    { 256ul, ngraph::Shape {}, { -128.f }, { 127.f } }, { 256ul, ngraph::Shape {6, 1, 1, 1}, { -128.f }, { 127.f } },
    // },
    //{
    //    { 256ul, ngraph::Shape {}, { 0.f }, { 255.f } }, { 256ul, ngraph::Shape {}, { -128.f }, { 127.f } },
    //},
    // { {}, {} }
};

INSTANTIATE_TEST_CASE_P(LPT, ConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::SizeVector({ 1, 3, 16, 16 })),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(versions),
        ::testing::ValuesIn(params)),
    ConvolutionTransformation::getTestCaseName);
}  // namespace
