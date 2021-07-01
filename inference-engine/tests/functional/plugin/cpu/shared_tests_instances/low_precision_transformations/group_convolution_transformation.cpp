// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/group_convolution_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ngraph::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(true),
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(false),
};

const std::vector<LayerTestsDefinitions::GroupConvolutionTransformationParam> params = {
    // group convolution, tensor quantization
    {
        ngraph::Shape{ 1, 6, 24, 24 },
        ngraph::Shape{ 1, 24, 18, 18 },
        3ul,
        -1,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        "Convolution",
        "U8"
    },
    // group convolution, tensor quantization
    {
        ngraph::Shape{ 1, 6, 24, 24 },
        ngraph::Shape{ 1, 24, 18, 18 },
        3ul,
        0,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        "Convolution",
        "U8"
    },
    // group convolution, tensor quantization
    {
        ngraph::Shape{ 1, 6, 24, 24 },
        ngraph::Shape{ 1, 24, 18, 18 },
        3ul,
        1,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        "Convolution",
        "U8"
    },
    // group convolution, per-channel quantization
    {
        ngraph::Shape{ 1, 6, 24, 24 },
        ngraph::Shape{ 1, 24, 18, 18 },
        3ul,
        -1,
        {
            256ul,
            ngraph::Shape { 6, 1, 1, 1 },
            { 0.f },
            { 25.5f },
            { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f },
            { 25.5f, 25.5f, 25.5f / 2.f, 25.5f / 2.f, 25.5f / 4.f, 25.5f / 4.f }
        },
        { 255ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
    },
    // depth-wise convolution, tensor quantization
    {
        ngraph::Shape{ 1, 6, 24, 24 },
        ngraph::Shape{ 1, 6, 18, 18 },
        6ul,
        -1,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
    },
    // depth-wise convolution, per-channel quantization
    {
        ngraph::Shape{ 1, 6, 24, 24 },
        ngraph::Shape{ 1, 6, 18, 18 },
        6ul,
        -1,
        {
            256ul,
            ngraph::Shape { 6, 1, 1, 1 },
            { 0.f },
            { 25.5f },
            { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f },
            { 25.5f, 25.5f, 25.5f / 2.f, 25.5f / 2.f, 25.5f / 4.f, 25.5f / 4.f }
        },
        { 255ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, GroupConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    GroupConvolutionTransformation::getTestCaseName);
}  // namespace
