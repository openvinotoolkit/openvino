// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/reshape_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams(),
};

const std::vector<ReshapeTransformationParam> params = {
    // 3D -> 4D
    {
        { 1, 3, 32 },
        { 1, 3, 4, 8 },
        { 256ul, ov::Shape{ 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        "Reshape",
        "U8"
    },
    // 3D -> 1D
    {
        { 1, 3, 32 },
        { -1 },
        { 256ul, ov::Shape{}, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        "Reshape",
        "U8"
    },
    // 4D -> 3D
    {
        { 1, 3, 16, 16 },
        { 1, 3, 256 },
        { 256ul, ov::Shape{ 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        "Reshape",
        "U8"
    },
    // 4D -> 3D
    {
        { 1, 3, 16, 16 },
        { 0, 3, -1 },
        { 256ul, ov::Shape{ 1, 3, 1, 1 }, { 0.f }, { 255.f }, { 0.f, 0.f, 0.f }, { 255.f, 25.5f, 2.55f } },
        "Reshape",
        "U8"
    },
    // 4D -> 2D
    {
        { 1, 3, 4, 8 },
        { 1, -1 },
        { 256ul, ov::Shape{ 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        "Reshape",
        "U8"
    },
    // 4D -> 6D
    {
        { 1, 3, 4, 8 },
        { 1, 3, 4, 8, 1, 1 },
        { 256ul, ov::Shape{ 1, 1, 1, 1}, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        "Reshape",
        "U8"
    },
    // 4D -> 2D
    {
        { 1, 3, 4, 8 },
        { 1, -1 },
        {
            256ul,
            ov::Shape{ 1, 3, 1, 1 },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f / 2.f, 255.f / 3.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f / 2.f, 255.f / 3.f },
        },
        "Reshape",
        "U8"
    },
    // 4D -> 3D
    {
        { 1, 3, 4, 8 },
        { 1, 3, -1 },
        {
            256ul,
            ov::Shape{ 1, 3, 1, 1 },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f / 2.f, 255.f / 3.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f / 2.f, 255.f / 3.f },
        },
        "Reshape",
        "U8"
    },
    // per-channel
    // 4D -> 3D
    {
        { 1, 3, 4, 8 },
        { 1, -1, 8 },
        {
            256ul,
            ov::Shape{ 1, 3, 1, 1 },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f / 2.f, 255.f / 3.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f / 2.f, 255.f / 3.f },
        },
        "Reshape",
        "U8"
    },
    // Channels count reducing, per-channel dequantizations 4d -> 4d
    {
        { 1, 3, 16, 16 },
        { 1, 1, 48, 16 },
        { 256ul, ov::Shape{ 1, 3, 1, 1 },
          { 0.f, 0.f, 0.f }, { 255.f, 255.f, 255.f },
          { 0.f, 0.f, 0.f }, { 255.f, 25.5f, 2.55f } },
        "Reshape",
        "FP32"
    },
    // Channels count reducing, per-channel dequantizations 3d -> 4d
    {
        { 1, 3,  16 },
        { 1, 1, 6, 8 },
        { 256ul, ov::Shape{ 1, 3, 1 },
                { 0.f, 0.f, 0.f }, { 255.f, 255.f, 255.f },
                { 0.f, 0.f, 0.f }, { 255.f, 25.5f, 2.55f } },
        "Reshape",
        "FP32"
    },
    // Channels count reducing, per-channel dequantizations 4d -> 3d
    {
        { 1, 3, 2, 4 },
        { 1, 1, 24 },
        { 256ul, ov::Shape{ 1, 3, 1, 1 },
                { 0.f, 0.f, 0.f }, { 255.f, 255.f, 255.f },
                { 0.f, 0.f, 0.f }, { 255.f, 25.5f, 2.55f } },
        "Reshape",
        "FP32"
    },
    // Channels count reducing, per-channel dequantizations 5d -> 3d
    {
        { 1, 3, 2, 4, 2 },
        { 1, 1, 48 },
        { 256ul, ov::Shape{ 1, 3, 1, 1, 1 },
                { 0.f, 0.f, 0.f }, { 255.f, 255.f, 255.f },
                { 0.f, 0.f, 0.f }, { 255.f, 25.5f, 2.55f } },
        "Reshape",
        "FP32"
    },
    // Channels count reducing, per-channel dequantizations 5d -> 4d
    {
        { 1, 3, 2, 4, 2 },
        { 1, 1, 3, 16 },
        { 256ul, ov::Shape{ 1, 3, 1, 1, 1 },
                { 0.f, 0.f, 0.f }, { 255.f, 255.f, 255.f },
                { 0.f, 0.f, 0.f }, { 255.f, 25.5f, 2.55f } },
        "Reshape",
        "FP32"
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ReshapeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    ReshapeTransformation::getTestCaseName);
}  // namespace




