// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <gtest/gtest.h>

#include "low_precision_transformations/reduce_min_transformation.hpp"


using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ngraph::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
     LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

const std::vector<LayerTestsDefinitions::ReduceMinTransformationParam> params = {
    {
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 127.f } },
        { 2, 3 },
        true,
        "Output_original",
        "U8"
    },
    {
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 127.f } },
        { 2, 3 },
        false,
        "Output_original",
        "U8"
    },
    {
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 127.f } },
        { 1 },
        true,
        "Output_original",
        "U8"
    },
    {
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 127.f } },
        { 1 },
        false,
        "Output_original",
        "U8"
    },
    {
        {
            256ul, ngraph::Shape{ 1, 3, 1, 1 },
            { -127.f, -127.f, -127.f },
            { 128.f, 128.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 25.5f, 2.55f }
        },
        { 2, 3 },
        true,
        "Output_original",
        "U8"
    },
    {
        {
            256ul, ngraph::Shape{ 1, 3, 1, 1 },
            { -127.f, -127.f, -127.f },
            { 128.f, 128.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 25.5f, 2.55f }
        },
        { 2, 3 },
        false,
        "Output_original",
        "U8"
    },
    {
        {
            256ul, ngraph::Shape{ 1, 3, 1, 1 },
            { -127.f, -127.f, -127.f },
            { 128.f, 128.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 25.5f, 2.55f }
        },
        { 0, 1 },
        true,
        "Output",
        "FP32"
    },
    {
        {
            256ul, ngraph::Shape{ 1, 3, 1, 1 },
            { -127.f, -127.f, -127.f },
            { 128.f, 128.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 25.5f, 2.55f }
        },
        { 0, 1 },
        false,
        "Output",
        "FP32"
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ReduceMinTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ngraph::Shape({ 1, 3, 10, 10 })),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    ReduceMinTransformation::getTestCaseName);

}  // namespace



