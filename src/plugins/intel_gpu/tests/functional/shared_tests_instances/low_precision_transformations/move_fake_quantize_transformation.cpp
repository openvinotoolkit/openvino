// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/move_fake_quantize_transformation.hpp"
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

const std::vector<LayerTestsDefinitions::MoveFakeQuantizeTransformationParam> params = {
    // without operation
    {
        3,
        "",
        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
        {},
        {},
        "Concat",
        "U8",
        1,
    },
    // with ReLU operation
    {
        3,
        "relu",
        { 256ul, {}, { -12.7f }, { 12.7f }, { -12.7f }, { 12.7f }},
        {},
        {},
        "Concat",
        "U8",
        1
    },
    // negative axis
    {
        3,
        "",
        {256ul, {},  {-1.28f}, {1.27f}, {-1.28f}, {1.27f}},
        {},
        {},
        "Concat",
        "FP32",
        0
    },
    // Q/DQ
    {
        3,
        "",
        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
        { ov::element::u8 },
        {
            { ov::element::f32 },
            {},
            { 0.01f }
        },
        "Concat",
        "U8",
        1
    },
    // Q/DQ with ReLU
    {
        3,
        "relu",
        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
        { ov::element::u8 },
        {
            { ov::element::f32 },
            {},
            { 0.01f }
        },
        "Concat",
        "U8",
        1
    },
    // multi chanel
    {
        3,
        "relu",
        {   256ul,
            {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
            {-2.66068696975708f}, {2.6399004459381104f},
            {-31.695816040039062f, -35.69844055175781f, -49.126914978027344f},
            {277.8320007324219f, 267.07110595703125f, 254.99429321289062f}
        },
        {},
        {},
        "Concat",
        "U8",
        1
    },
    // Q/DQ with multi-channels
    {
        3,
        "",
        {
            256ul,
            {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
            {0.f, 0.f, 0.f},
            {2.55f, 2.55f, 2.55f},
            {0.f, 0.f, 0.f},
            {255.f, 255.f, 255.f}
        },
        { ov::element::u8 },
        {
            { ov::element::f32 },
            {},
            { {0.01f, 0.01f, 0.01f}, ov::element::f32, {1, 3, 1, 1} }
        },
        "Concat",
        "U8",
        1
    },
    // Q/DQ with multi-channels subtruct
    {
        3,
        "",
        {
            256ul,
            {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
            {0.f, 0.f, 0.f},
            {2.55f, 2.55f, 2.55f},
            {0.f, 0.f, 0.f},
            {255.f, 255.f, 255.f}
        },
        { ov::element::u8 },
        {
            { ov::element::f32 },
            { {0.01f, 0.01f, 0.01f}, ov::element::f32, {1, 3, 1, 1} },
            { 0.01f }
        },
        "Concat",
        "U8",
        1
    },
};

const std::vector<std::vector<ov::PartialShape>> shapes = {
    {{ 1, 1, 16, 16 }, { 1, 1, 16, 16 }, { 1, 1, 16, 16 }},
    {{ 4, 1, 16, 16 }, { 4, 1, 16, 16 }, { 4, 1, 16, 16 }}
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, MoveFakeQuantizeTransformation,
::testing::Combine(
    ::testing::ValuesIn(netPrecisions),
    ::testing::ValuesIn(shapes),
    ::testing::Values(ov::test::utils::DEVICE_GPU),
    ::testing::ValuesIn(trasformationParamValues),
    ::testing::ValuesIn({false, true}),
    ::testing::ValuesIn(params)),
MoveFakeQuantizeTransformation::getTestCaseName);
}  // namespace
