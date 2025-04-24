// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <gtest/gtest.h>

#include "low_precision_transformations/reduce_mean_transformation.hpp"


using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
     LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

const std::vector<LayerTestsDefinitions::ReduceMeanTransformationParam> params = {
    {
        { 256ul, ov::Shape{ 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 127.f } },
        {},
        {},
        {{2, 3}, true},
        {}
    },
    {
        { 256ul, ov::Shape{ 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 127.f } },
        {},
        {},
        {{2, 3}, false},
        {}
    },
    {
        { 256ul, ov::Shape{ 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 127.f } },
        {},
        {},
        {{ 1 }, true},
        {}
    },
    {
        { 256ul, ov::Shape{ 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 127.f } },
        {},
        {},
        {{ 1 }, false},
        {}
    },
    {
        {
            256ul, ov::Shape{ 1, 3, 1, 1 },
            { -127.f, -127.f, -127.f },
            { 128.f, 128.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 25.5f, 2.55f }
        },
        {},
        {},
        {{ 2, 3 }, true},
        {}
    },
    {
        {
            256ul, ov::Shape{ 1, 3, 1, 1 },
            { -127.f, -127.f, -127.f },
            { 128.f, 128.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 25.5f, 2.55f }
        },
        {},
        {},
        {{ 2, 3 }, false},
        {}
    },
    {
        {
            256ul, ov::Shape{ 1, 3, 1, 1 },
            { -127.f, -127.f, -127.f },
            { 128.f, 128.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 25.5f, 2.55f }
        },
        {},
        {},
        {{ 0, 1 }, true},
        {}
    },
    {
        {
            256ul, ov::Shape{ 1, 3, 1, 1 },
            { -127.f, -127.f, -127.f },
            { 128.f, 128.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 25.5f, 2.55f }
        },
        {},
        {},
        {{ 0, 1 }, false},
        {}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ReduceMeanTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::PartialShape({ 1, 3, 10, 10 })),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    ReduceMeanTransformation::getTestCaseName);

}  // namespace



