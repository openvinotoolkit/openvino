// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <gtest/gtest.h>

#include "low_precision_transformations/split_transformation.hpp"
#include "common_test_utils/test_constants.hpp"


using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<ngraph::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams(),
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(false),
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsI8I8(),
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

const std::vector<LayerTestsDefinitions::SplitTransformationParam> params = {
    // tensor quantization, split second dimension
    {
        { 256ul, ngraph::Shape{ }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f / 2.f } },
        2, 2ul
    },
    // tensor quantization, split third dimension
    {
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { -12.8f }, { 12.7f }, { 0.f }, { 25.5f } },
        -1, 2ul
    },
    // per-channel quantization with the same values, split second dimension
    {
        {
            256ul, ngraph::Shape{ 1, 3, 1, 1 },
            { -127.f, -127.f, -127.f },
            { 128.f, 128.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f, 255.f }
        },
        2, 4ul
    },
    // per-channel quantization with the same values, per-channel split
    {
        {
            256ul, ngraph::Shape{ 1, 3, 1, 1 },
            { -127.f, -127.f, -127.f },
            { 128.f, 128.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f, 255.f }
        },
        1, 3ul
    },
    // per-channel quantization with different values, split third dimension
    {
        {
            256ul,
            ngraph::Shape{ 1, 3, 1, 1 },
            { -127.f, 0.f, 128.f / 2.f },
            { 128.f / 4.f, 128.f / 2.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f / 4.f, 255.f / 2.f, 255.f }
        },
        -1, 4ul
    },
    // per-channel quantization with different values, per-channel split
    {
        {
            256ul,
            ngraph::Shape{ 1, 3, 1, 1 },
            { -127.f, 0.f, 128.f / 2.f },
            { 128.f / 4.f, 128.f / 2.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f / 4.f, 255.f / 2.f, 255.f }
        },
        1, 3ul
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, SplitTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ngraph::PartialShape({ 1, 3, 16, 16 })),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    SplitTransformation::getTestCaseName);
}  // namespace
