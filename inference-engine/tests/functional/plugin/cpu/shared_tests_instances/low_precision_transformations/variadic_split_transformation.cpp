// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <gtest/gtest.h>

#include "low_precision_transformations/variadic_split_transformation.hpp"
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
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsI8I8(),
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

const std::vector<LayerTestsDefinitions::VariadicSplitTransformationParam> params{
    {
        { 256ul, ngraph::Shape{ }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 2 }, std::vector<size_t>{9, 7}
    },
    {
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { -12.8f }, { 12.7f }, { 0.f }, { 25.5f } },
        { -1 }, std::vector<size_t>{15, 1}
    },
    {
        {
            256ul, ngraph::Shape{ 1, 3, 1, 1 },
            { -127.f, -127.f, -127.f },
            { 128.f, 128.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f, 255.f }
        },
        { 1 }, std::vector<size_t>{1, 1, 1}
    },
    {
        {
            256ul, ngraph::Shape{ 1, 3, 1, 1 },
            { -127.f, -127.f, -127.f },
            { 128.f, 128.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f, 255.f }
        },
        { -1 }, std::vector<size_t>{4, 3, 2, 7}
    }
};

INSTANTIATE_TEST_CASE_P(LPT, VariadicSplitTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ngraph::Shape({ 1, 3, 16, 16 })),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::Values(LayerTestsUtils::LayerTransformation::LptVersion::nGraph),
        ::testing::ValuesIn(params)),
    VariadicSplitTransformation::getTestCaseName);

}  // namespace
