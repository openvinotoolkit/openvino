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
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(true)
};

const std::vector<LayerTestsDefinitions::ConvolutionBackpropDataTransformationParam> params = {
    // FQ on weights
    {
        {256ul, ngraph::Shape{1, 1, 1, 1}, { 0.f }, { 25.5f }, { 0.f }, { 25.5f }},
        {255ul, ngraph::Shape{1, 1, 1, 1}, { -12.7f }, { 12.7f }, { -12.7f }, { 12.7f }},
        "convolutionBackpropData_original",
        "U8"
    },
    // FQ on weights
    // with zero point
    {
        {256ul, ngraph::Shape{1, 1, 1, 1}, { 0.f }, { 255.f }, { -12.7f }, { 12.8f }},
        {255ul, ngraph::Shape{1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f }},
        "",
        ""
    },
    // without zero point
    {
        {256ul, ngraph::Shape{1, 1, 1, 1}, { 0.f }, { 255.f }, { 0.f }, { 25.5f }},
        {255ul, ngraph::Shape{1, 1, 1, 1}, { 0.f }, { 254.f }, { 0.f }, { 25.4f }},
        "",
        ""
    },
    // with incorrect zero point on activations
    {
        {256ul, ngraph::Shape{1, 1, 1, 1}, { 5.f }, { 6.f }, { 5.f }, { 6.f }},
        {255ul, ngraph::Shape{1, 1, 1, 1}, { 0.f }, { 254.f }, { 0.f }, { 25.4f }},
        "",
        ""
    },
    // with incorrect zero point on weights
    {
        {256ul, ngraph::Shape{1, 1, 1, 1}, { 0.f }, { 255.f }, { 0.f }, { 25.5f }},
        {255ul, ngraph::Shape{1, 1, 1, 1}, { 5.f }, { 6.f }, { 5.f }, { 6.f }},
        "",
        ""
    },
    // QDq on weights
    // with zero point
    {
        {256ul, ngraph::Shape{1, 1, 1, 1}, { 0.f }, { 255.f }, { -12.7f }, { 12.8f }},
        {{ngraph::element::f32}, { {12.f}, ngraph::element::f32, {}, false }, { {4.f}, ngraph::element::f32, {}, false }},
        "",
        ""
    },
    // without zero point
    {
        {256ul, ngraph::Shape{1, 1, 1, 1}, { 0.f }, { 255.f }, { 0.f }, { 25.5f }},
        {{ngraph::element::f32}, {}, { {4.f}, ngraph::element::f32, {}, false }},
        "",
        ""
    },
    // with incorrect zero point on activations
    {
        {256ul, ngraph::Shape{1, 1, 1, 1}, { 5.f }, { 6.f }, { 5.f }, { 6.f }},
        {{ngraph::element::f32}, { {12.f}, ngraph::element::f32, {}, false }, { {4.f}, ngraph::element::f32, {}, false }},
        "",
        ""
    },
    // with incorrect zero point on weights
    {
        {256ul, ngraph::Shape{1, 1, 1, 1}, { 0.f }, { 255.f }, { -12.7f }, { 12.8f }},
        {{ngraph::element::f32}, { {1000.f}, ngraph::element::f32, {}, false }, { {4.f}, ngraph::element::f32, {}, false }},
        "",
        ""
    },
    // issue #56886: with incorrect dequantization on weights
    {
        {256ul, ngraph::Shape{1, 1, 1, 1}, { 0.f }, { 255.f }, { 0.f }, { 25.5f }},
        {{ngraph::element::f32}, {}, { {4.f, 2.f, 4.f, 2.f, 4.f, 2.f, 4.f, 2.f}, ngraph::element::f32, {8, 1, 1, 1}, false }},
        "",
        ""
    }
};

const std::vector<std::pair<ngraph::Shape, bool>> inputShapes = {
    {{ 1, 8, 16, 16 }, true}
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
