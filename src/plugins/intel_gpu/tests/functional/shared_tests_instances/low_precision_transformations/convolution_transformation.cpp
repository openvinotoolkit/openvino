// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/convolution_transformation.hpp"
#include "low_precision_transformations/convolution_with_incorrect_weights.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams()
};

const std::vector<LayerTestsDefinitions::ConvolutionTransformationParam> params = {
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        false,
        {},
        false,
        "Convolution",
        "FP32"
    },
    {
        {},
        false,
        { 255ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -12.7f }, { 12.7f } },
        false,
        "Convolution",
        "FP32"
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        false,
        { 255ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -12.7f }, { 12.7f } },
        false,
        "Convolution",
        "U8"
    },
    {
        { 256ul, ngraph::Shape {}, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        false,
        { 255ul, ngraph::Shape {}, { 0.f }, { 254.f }, { -12.7f }, { 12.7f } },
        false,
        "Convolution",
        "U8"
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -12.75f }, { 6.375f } },
        true,
        { 255ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -12.7f }, { 12.7f } },
        false,
        "Convolution",
        "U8"
    },
    {
        { 256ul, ngraph::Shape { 1 }, { 0.f }, { 255.f }, { -18.7f }, { 18.8f } },
        true,
        {
            255ul, ngraph::Shape { 6, 1, 1, 1 }, { -0.6f }, { 0.6f },
            { -1.52806e-39f, -0.2, -0.3, -0.3, -0.2, -0.1 }, { 1.52806e-39f, 0.2, 0.3, 0.3, 0.2, 0.1 }
        },
        false,
        "Convolution",
        "U8"
    },
    {
        { 256ul, ngraph::Shape { 1 }, { 0.f }, { 255.f }, { -18.7f }, { 18.8f } },
        true,
        {
            255ul, ngraph::Shape { 6, 1, 1, 1 }, { -0.6f }, { 0.6f },
            { -1.52806e-39f, -1.52806e-39f, -1.52806e-39f, -1.52806e-39f, -1.52806e-39f, -1.52806e-39f },
            { 1.52806e-39f, 1.52806e-39f, 1.52806e-39f, 1.52806e-39f, 1.52806e-39f, 1.52806e-39f }
        },
        false,
        "Convolution",
        "U8"
    },
    // not supported quantization level on data
    {
        { 65536ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        false,
        { 255ul, ngraph::Shape{1, 1, 1, 1}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
        false,
        "Convolution",
        "FP32"
    },
    // not supported quantization level on data & weights
    {
        { 65536ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        false,
        { 65536ul, ngraph::Shape{1, 1, 1, 1}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
        false,
        "Convolution",
        "FP32"
    },
    // not supported quantization level on weights
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        false,
        { 65536ul, ngraph::Shape{1, 1, 1, 1}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
        false,
        "Convolution",
        "FP32"
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ngraph::Shape({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    ConvolutionTransformation::getTestCaseName);

const std::vector<LayerTestsDefinitions::ConvolutionWIthIncorrectWeightsParam> incorrectWeightsParams = {
    // incorrect weights
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        { 255ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        false
    },
    // correct weights
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        { 255ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        true
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ConvolutionWIthIncorrectWeightsTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ngraph::Shape({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(incorrectWeightsParams)),
    ConvolutionWIthIncorrectWeightsTransformation::getTestCaseName);
}  // namespace
