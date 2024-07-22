// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/convolution_backprop_data_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(true)
};

const std::vector<LayerTestsDefinitions::ConvolutionBackpropDataTransformationParam> params = {
    // FQ on weights
    {
        {256ul, ov::Shape{1, 1, 1, 1}, { 0.f }, { 25.5f }, { 0.f }, { 25.5f }},
        {255ul, ov::Shape{1, 1, 1, 1}, { -12.7f }, { 12.7f }, { -12.7f }, { 12.7f }},
        "convolutionBackpropData_original",
        "u8"
    },
    // FQ on weights
    {
        {256ul, ov::Shape{}, { 0.f }, { 25.5f }, { 0.f }, { 25.5f }},
        {255ul, ov::Shape{}, { -12.7f }, { 12.7f }, { -12.7f }, { 12.7f }},
        "convolutionBackpropData_original",
        "u8"
    },
    // FQ on weights
    {
        {256ul, ov::Shape{1, 1, 1, 1}, { -12.8f }, { 12.7f }, { -12.8f }, { 12.7f }},
        {255ul, ov::Shape{1, 1, 1, 1}, { -12.7f }, { 12.7f }, { -12.7f }, { 12.7f }},
        "convolutionBackpropData_original",
        "i8"
    },
    // FQ on weights
    // with zero point
    {
        {256ul, ov::Shape{1, 1, 1, 1}, { 0.f }, { 255.f }, { -12.7f }, { 12.8f }},
        {255ul, ov::Shape{1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f }},
        "",
        ""
    },
    // without zero point
    {
        {256ul, ov::Shape{1, 1, 1, 1}, { 0.f }, { 255.f }, { 0.f }, { 25.5f }},
        {255ul, ov::Shape{1, 1, 1, 1}, { 0.f }, { 254.f }, { 0.f }, { 25.4f }},
        "",
        ""
    },
    // TODO: check fails in CI
//    // with incorrect zero point on activations
//    {
//        {256ul, ov::Shape{1, 1, 1, 1}, { 5.f }, { 6.f }, { 5.f }, { 6.f }},
//        {255ul, ov::Shape{1, 1, 1, 1}, { 0.f }, { 254.f }, { 0.f }, { 25.4f }},
//        "",
//        ""
//    },
//    // with incorrect zero point on weights
//    {
//        {256ul, ov::Shape{1, 1, 1, 1}, { 0.f }, { 255.f }, { 0.f }, { 25.5f }},
//        {255ul, ov::Shape{1, 1, 1, 1}, { 5.f }, { 6.f }, { 5.f }, { 6.f }},
//        "",
//        ""
//    },
    // QDq on weights
    // with zero point
    {
        {256ul, ov::Shape{1, 1, 1, 1}, { 0.f }, { 255.f }, { -12.7f }, { 12.8f }},
        {{ov::element::f32}, { {12.f}, ov::element::f32, {}, false }, { {4.f}, ov::element::f32, {}, false }},
        "",
        ""
    },
    // without zero point
    {
        {256ul, ov::Shape{1, 1, 1, 1}, { 0.f }, { 255.f }, { 0.f }, { 25.5f }},
        {{ov::element::f32}, {}, { {4.f}, ov::element::f32, {}, false }},
        "",
        ""
    },
    // with incorrect zero point on activations
    {
        {256ul, ov::Shape{1, 1, 1, 1}, { 5.f }, { 6.f }, { 5.f }, { 6.f }},
        {{ov::element::f32}, { {12.f}, ov::element::f32, {}, false }, { {4.f}, ov::element::f32, {}, false }},
        "",
        ""
    },
    // with incorrect zero point on weights
    {
        {256ul, ov::Shape{1, 1, 1, 1}, { 0.f }, { 255.f }, { -12.7f }, { 12.8f }},
        {{ov::element::f32}, { {1000.f}, ov::element::f32, {}, false }, { {4.f}, ov::element::f32, {}, false }},
        "",
        ""
    }
};

const std::vector<std::pair<ov::PartialShape, bool>> inputShapes_4D = {
        {{ 1, 8, 16, 16 }, false},
        {{ 1, 32, 16, 16 }, true}
};

const std::vector<ov::Shape> outputShapes_4D = {
        { 16, 16 }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT_4D, ConvolutionBackpropDataTransformation,
    ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::ValuesIn(inputShapes_4D),
            ::testing::ValuesIn(outputShapes_4D),
            ::testing::Values(ov::test::utils::DEVICE_GPU),
            ::testing::ValuesIn(trasformationParamValues),
            ::testing::ValuesIn(params)),
    ConvolutionBackpropDataTransformation::getTestCaseName);

const std::vector<std::pair<ov::PartialShape, bool>> inputShapes_3D = {
    {{ 1, 32, 16, 16 }, true}
};

const std::vector<ov::Shape> outputShapes_3D = {
    { 16 }
};

const std::vector<LayerTestsDefinitions::ConvolutionBackpropDataTransformationParam> params_3D = {
    // FQ on weights
    {
        {256ul, ov::Shape{1, 1, 1}, { 0.f }, { 25.5f }, { 0.f }, { 25.5f }},
        {255ul, ov::Shape{1, 1, 1}, { -12.7f }, { 12.7f }, { -12.7f }, { 12.7f }},
        "convolutionBackpropData_original",
        "u8"
    },
    // Qdq on weights
    {
        {256ul, ov::Shape{1, 1, 1}, { 0.f }, { 255.f }, { 0.f }, { 25.5f }},
        {{ov::element::f32}, {}, { {4.f}, ov::element::f32, {}, false }},
        "convolutionBackpropData_original",
        "u8"
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT_3D, ConvolutionBackpropDataTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputShapes_3D),
        ::testing::ValuesIn(outputShapes_3D),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params_3D)),
    ConvolutionBackpropDataTransformation::getTestCaseName);
}  // namespace
