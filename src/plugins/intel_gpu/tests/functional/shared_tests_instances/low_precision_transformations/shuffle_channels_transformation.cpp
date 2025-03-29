// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/shuffle_channels_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<ov::PartialShape> inputShapes = {
    { 1, 3, 16, 16 }
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams(),
};

const std::vector<LayerTestsDefinitions::ShuffleChannelsTransformationParam> params = {
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        0,
        1,
    },
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        -3,
        1,
    },
    {
        {
            256ul,
            ov::Shape { 1, 3, 1, 1 },
            { 0.f },
            { 25.5f },
            { 0.f, 0.f, 0.f },
            { 25.5f / 2.f, 25.5f / 4.f, 25.5f }
        },
        -3,
        1,
    },
    {
        {
            256ul,
            ov::Shape { 1, 3, 1, 1 },
            { 0.f },
            { 25.5f },
            { -4.f, -3.f, 0.f },
            { 10.f, 12.f, 25.5f }
        },
        -3,
        1,
    },
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        2,
        4,
    },
    {
        {
            256ul,
            ov::Shape { 1, 3, 1, 1 },
            { 0.f },
            { 25.5f },
            { 0.f, 0.f, 0.f },
            { 25.5f / 2.f, 25.5f / 4.f, 25.5f }
        },
        -1,
        8,
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ShuffleChannelsTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    ShuffleChannelsTransformation::getTestCaseName);
}  // namespace
