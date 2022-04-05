// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include "low_precision_transformations/tile_transformation.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32
};

const std::vector<ngraph::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams(),
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(false),
};

const std::vector<LayerTestsDefinitions::TileTransformationParam> params{
    // u8
    {
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
    },
    // i8
    {
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { -12.8f }, { 12.7f } },
    },
    // per-channel quantization with the same values
    {
        {
            256ul,
            ngraph::Shape{ 1, 3, 1, 1 },
            { -127.f, -127.f, -127.f },
            { 128.f, 128.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f, 255.f }
        },
    },
    // per-channel quantization with different values
    {
        {
            256ul,
            ngraph::Shape{ 1, 3, 1, 1 },
            { -127.f, 0.f, 128.f / 2.f },
            { 128.f / 4.f, 128.f / 2.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f / 4.f, 255.f / 2.f, 255.f }
        },
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, TileTransformation,
    ::testing::Combine(
         ::testing::ValuesIn(netPrecisions),
         ::testing::Values(ngraph::PartialShape({ 1, 3, 16, 16 })),
         ::testing::Values(CommonTestUtils::DEVICE_CPU),
         ::testing::ValuesIn(trasformationParamValues),
         ::testing::ValuesIn(params)),
    TileTransformation::getTestCaseName);
}  // namespace
