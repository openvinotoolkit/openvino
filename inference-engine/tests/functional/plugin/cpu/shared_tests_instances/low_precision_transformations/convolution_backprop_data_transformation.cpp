// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/convolution_backprop_data_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32
};

const std::vector<ngraph::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(true),
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(false)
};

const std::vector<LayerTestsDefinitions::ConvolutionBackpropDataTransformationParam> params = {
    {
        {256ul, ngraph::Shape{1, 1, 1, 1}, { 0.f }, { 255.f }, { 0.f }, { 25.5f }},
        {255ul, ngraph::Shape{1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f }},
        "",
        ""
    },
    {
        {256ul, ngraph::Shape{1, 1, 1, 1}, { 0.f }, { 255.f }, { 0.f }, { 25.5f }},
        {255ul, ngraph::Shape{1, 1, 1, 1}, { 0.f }, { 254.f }, { 0.f }, { 25.4f }},
        "",
        ""
    },
    {
        {256ul, ngraph::Shape{1, 1, 1, 1}, { 0.f }, { 255.f }, { 0.f }, { 25.5f }},
        {ngraph::element::f32, { 12.f }, { 4.f }},
        "",
        ""
    }
};

const std::vector<ngraph::Shape> inputShapes = {
    { 1, 8, 16, 16 }
};

const std::vector<ngraph::Shape> outputShapes = {
    { 16, 16 }
};

INSTANTIATE_TEST_CASE_P(smoke_LPT, ConvolutionBackpropDataTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(outputShapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    ConvolutionBackpropDataTransformation::getTestCaseName);
}  // namespace
