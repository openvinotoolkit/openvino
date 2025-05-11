// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <gtest/gtest.h>

#include "low_precision_transformations/clamp_transformation.hpp"


using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
     LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams(),
    //  LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(false),
    //  LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsI8I8(),
    //  LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

const std::vector<LayerTestsDefinitions::ClampTransformationParam> params = {
    // tensor quantization
    {
        { 256ul, ov::Shape{ 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 127.f } },
        {
            {},
            {{0.f, 0.f, 0.f}},
            {{0.5f, 0.5f, 0.5f}}
        },
        0.0,
        127.0
    },
    // tensor quantization
    {
        { 256ul, ov::Shape{ 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { -12.8f }, { 12.7f } },
        {
            {},
            {{0.f, 0.f, 0.f}},
            {{0.1f, 0.1f, 0.1f}}
        },
        0.0,
        255.0
    },
    // per-channel quantization with the same values
    {
        {
            256ul, ov::Shape{ 1, 3, 1, 1 },
            { -127.f, -127.f, -127.f },
            { 128.f, 128.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f, 255.f }
        },
        {},
        0.0,
        255.0
    },
    // per-channel quantization with different values
    {
        {
            256ul,
            ov::Shape{ 1, 3, 1, 1 },
            { -127.f, 0.f, 128.f / 2.f },
            { 128.f / 4.f, 128.f / 2.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f / 4.f, 255.f / 2.f, 255.f }
        },
        {},
        0.0,
        128.0
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ClampTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::PartialShape({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    ClampTransformation::getTestCaseName);

}  // namespace



