// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/group_convolution_transformation.hpp"
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

const std::vector<bool> addPrecisionPreserved = { true, false };

const std::vector<std::pair<ov::PartialShape, ov::Shape>> inputShapes = {
    {{ 1, 6, 24, 24 }, { 1, 24, 18, 18 }}
};

const std::vector<LayerTestsDefinitions::GroupConvolutionTransformationParam> params = {
    // group convolution, tensor quantization
    {
        3ul,
        -1,
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        true,
        "Convolution",
        "U8"
    },
    // group convolution, tensor quantization
    {
        3ul,
        0,
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        true,
        "Convolution",
        "U8"
    },
    // group convolution, tensor quantization
    {
        3ul,
        1,
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        true,
        "Convolution",
        "U8"
    },
    // group convolution, per-channel quantization
    {
        3ul,
        -1,
        {
            256ul,
            {/* will be filled in automatically */},
            { 0.f },
            { 25.5f },
            { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f },
            { 25.5f, 25.5f, 25.5f / 2.f, 25.5f / 2.f, 25.5f / 4.f, 25.5f / 4.f }
        },
        { 255ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        true,
    },
    // group convolution, per-channel weights quantization
    {
        3ul,
        -1,
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        false,
        "",
        ""
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, GroupConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(params),
        ::testing::ValuesIn(addPrecisionPreserved)),
    GroupConvolutionTransformation::getTestCaseName);

namespace test_values_4d {
const std::vector<std::pair<ov::PartialShape, ov::Shape>> inputShapes = {
    {{ 1, 6, 24, 24 }, { 1, 24, 18, 18 }},
};

const std::vector<LayerTestsDefinitions::GroupConvolutionTransformationParam> params = {
    // group convolution without reshape, per channel quantization
    {
        3ul,
        -1,
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, ov::Shape { 3, 8, 1, 1, 1 }, { -127.f }, { 127.f }, { -127.f }, { 127.f } },
        false,
        "Convolution",
        "U8"
    },
    // group convolution without reshape, per channel quantization with different values
    {
        3ul,
        -1,
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, ov::Shape { 3, 8, 1, 1, 1 },
            {-127.f, -12.7f, -1.27f, -127.f, -12.7f, -1.27f, -127.f, -12.7f,
             -127.f, -12.7f, -1.27f, -127.f, -12.7f, -1.27f, -127.f, -12.7f,
             -127.f, -12.7f, -1.27f, -127.f, -12.7f, -1.27f, -127.f, -12.7f},
            {127.f, 12.7f, 1.27f, 127.f, 12.7f, 1.27f, 127.f, 12.7f,
             127.f, 12.7f, 1.27f, 127.f, 12.7f, 1.27f, 127.f, 12.7f,
             127.f, 12.7f, 1.27f, 127.f, 12.7f, 1.27f, 127.f, 12.7f},
            {-127.f, -12.7f, -1.27f, -127.f, -12.7f, -1.27f, -127.f, -12.7f,
             -127.f, -12.7f, -1.27f, -127.f, -12.7f, -1.27f, -127.f, -12.7f,
             -127.f, -12.7f, -1.27f, -127.f, -12.7f, -1.27f, -127.f, -12.7f},
            {127.f, 12.7f, 1.27f, 127.f, 12.7f, 1.27f, 127.f, 12.7f,
             127.f, 12.7f, 1.27f, 127.f, 12.7f, 1.27f, 127.f, 12.7f,
             127.f, 12.7f, 1.27f, 127.f, 12.7f, 1.27f, 127.f, 12.7f}
        },
        false,
        "Convolution",
        "U8"
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, GroupConvolutionTransformation,
     ::testing::Combine(
         ::testing::ValuesIn(netPrecisions),
         ::testing::Values(ov::test::utils::DEVICE_GPU),
         ::testing::ValuesIn(trasformationParamValues),
         ::testing::ValuesIn(inputShapes),
         ::testing::ValuesIn(params),
         ::testing::Values(false)),
         GroupConvolutionTransformation::getTestCaseName);
}  // namespace test_values_4d

namespace test_values_3d {
const std::vector<std::pair<ov::PartialShape, ov::Shape>> inputShapes = {
    {{ 1, 6, 24 }, { 1, 24, 18 }},
};

const std::vector<LayerTestsDefinitions::GroupConvolutionTransformationParam> params = {
    // group convolution without reshape, per channel quantization
    {
        3ul,
        -1,
        { 256ul, ov::Shape { 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, ov::Shape { 3, 8, 1, 1 }, { -127.f }, { 127.f }, { -127.f }, { 127.f } },
        false,
        "Convolution",
        "U8"
    },
    // group convolution without reshape, per channel quantization with different values
    {
        3ul,
        -1,
        { 256ul, ov::Shape { 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, ov::Shape { 3, 8, 1, 1 },
            {-127.f, -12.7f, -1.27f, -127.f, -12.7f, -1.27f, -127.f, -12.7f,
             -127.f, -12.7f, -1.27f, -127.f, -12.7f, -1.27f, -127.f, -12.7f,
             -127.f, -12.7f, -1.27f, -127.f, -12.7f, -1.27f, -127.f, -12.7f},
            {127.f, 12.7f, 1.27f, 127.f, 12.7f, 1.27f, 127.f, 12.7f,
             127.f, 12.7f, 1.27f, 127.f, 12.7f, 1.27f, 127.f, 12.7f,
             127.f, 12.7f, 1.27f, 127.f, 12.7f, 1.27f, 127.f, 12.7f},
            {-127.f, -12.7f, -1.27f, -127.f, -12.7f, -1.27f, -127.f, -12.7f,
             -127.f, -12.7f, -1.27f, -127.f, -12.7f, -1.27f, -127.f, -12.7f,
             -127.f, -12.7f, -1.27f, -127.f, -12.7f, -1.27f, -127.f, -12.7f},
            {127.f, 12.7f, 1.27f, 127.f, 12.7f, 1.27f, 127.f, 12.7f,
             127.f, 12.7f, 1.27f, 127.f, 12.7f, 1.27f, 127.f, 12.7f,
             127.f, 12.7f, 1.27f, 127.f, 12.7f, 1.27f, 127.f, 12.7f}
        },
        false,
        "Convolution",
        "U8"
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, GroupConvolutionTransformation,
     ::testing::Combine(
         ::testing::ValuesIn(netPrecisions),
         ::testing::Values(ov::test::utils::DEVICE_GPU),
         ::testing::ValuesIn(trasformationParamValues),
         ::testing::ValuesIn(inputShapes),
         ::testing::ValuesIn(params),
         ::testing::Values(false)),
         GroupConvolutionTransformation::getTestCaseName);
}  // namespace test_values_3d

namespace depthwise {
const std::vector<std::pair<ov::PartialShape, ov::Shape>> inputShapes = {
    {{ 1, 6, 24, 24 }, { 1, 6, 18, 18 }}
};

const std::vector<LayerTestsDefinitions::GroupConvolutionTransformationParam> params = {
    // depthwise convolution, tensor quantization
    {
        6ul,
        -1,
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        true,
    },
    // depthwise convolution, per-channel quantization
    {
        6ul,
        -1,
        {
            256ul,
            {/* will be filled in automatically */},
            { 0.f },
            { 25.5f },
            { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f },
            { 25.5f, 25.5f, 25.5f / 2.f, 25.5f / 2.f, 25.5f / 4.f, 25.5f / 4.f }
        },
        { 255ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        true,
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, GroupConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(params),
        ::testing::ValuesIn(addPrecisionPreserved)),
    GroupConvolutionTransformation::getTestCaseName);
}  // namespace depthwise
}  // namespace
