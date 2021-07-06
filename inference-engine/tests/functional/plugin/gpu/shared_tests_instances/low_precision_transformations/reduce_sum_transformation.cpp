// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <gtest/gtest.h>

#include "low_precision_transformations/reduce_sum_transformation.hpp"


using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<ngraph::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
     LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

const std::vector<LayerTestsDefinitions::ReduceSumTransformationParam> params = {
    {
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 127.f } },
        { 2, 3 },
        true,
    },
    {
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { 2.f }, { 10.f }, { 2.f }, { 10.f } },
        { 2, 3 },
        false,
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
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ReduceSumTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ngraph::Shape({ 1, 3, 10, 10 })),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    ReduceSumTransformation::getTestCaseName);

}  // namespace



