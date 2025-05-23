// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/group_convolution_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams()
};

const std::vector<bool> addPrecisionPreserved = { true, false };

const std::vector<std::pair<ov::PartialShape, ov::Shape>> inputShapes = {
    {{ 1, 6, 24, 24 }, { 1, 24, 18, 18 }},
    {{ 1, 6, 24 }, { 1, 24, 18 }}
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
        "u8"
    },
    // group convolution, tensor quantization
    {
        3ul,
        0,
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        true,
        "Convolution",
        "u8"
    },
    // group convolution, tensor quantization
    {
        3ul,
        1,
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        true,
        "Convolution",
        "u8"
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
        "Convolution",
        "u8"
    },
    // group convolution without reshape, tensor quantization
    {
        3ul,
        -1,
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        false,
        "Convolution",
        "u8"
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, GroupConvolutionTransformation,
     ::testing::Combine(
         ::testing::ValuesIn(netPrecisions),
         ::testing::Values(ov::test::utils::DEVICE_CPU),
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
        "u8"
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
        "u8"
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, GroupConvolutionTransformation,
     ::testing::Combine(
         ::testing::ValuesIn(netPrecisions),
         ::testing::Values(ov::test::utils::DEVICE_CPU),
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
        "u8"
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
        "u8"
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, GroupConvolutionTransformation,
     ::testing::Combine(
         ::testing::ValuesIn(netPrecisions),
         ::testing::Values(ov::test::utils::DEVICE_CPU),
         ::testing::ValuesIn(trasformationParamValues),
         ::testing::ValuesIn(inputShapes),
         ::testing::ValuesIn(params),
         ::testing::Values(false)),
         GroupConvolutionTransformation::getTestCaseName);
}  // namespace test_values_3d

namespace depthwise {
const std::vector<std::pair<ov::PartialShape, ov::Shape>> inputShapes = {
    {{ 1, 6, 24, 24 }, { 1, 6, 18, 18 }},
    {{ 1, 6, 24 }, { 1, 6, 18 }},
};

const std::vector<LayerTestsDefinitions::GroupConvolutionTransformationParam> params = {
    // depth-wise convolution, tensor quantization
    {
        6ul,
        -1,
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        true,
        "",
        ""
    },
    // depth-wise convolution, per-channel quantization
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
        "",
        ""
    },
    // depth-wise convolution, per-channel quantization
    {
        6ul,
        -1,
        {
            256ul,
            {/* will be filled in automatically */},
            { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f },
            { 25.5f, 25.5f, 25.5f / 2.f, 25.5f / 2.f, 25.5f / 4.f, 25.5f / 4.f },
            { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f },
            { 25.5f, 25.5f, 25.5f / 2.f, 25.5f / 2.f, 25.5f / 4.f, 25.5f / 4.f }
        },
        { 255ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        true,
        "",
        ""
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, GroupConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(params),
        ::testing::ValuesIn(addPrecisionPreserved)),
    GroupConvolutionTransformation::getTestCaseName);
} // namespace depthwise

namespace i8_3d {
const std::vector<std::pair<ov::PartialShape, ov::Shape>> inputShapes = {
    {{1, 6, 1, 24, 24}, {1, 24, 1, 18, 18}},
    {{1, 24, 8, 12, 12}, {1, 24, 1, 1, 1}}
};

const std::vector<LayerTestsDefinitions::GroupConvolutionTransformationParam> params = {
    // group convolution, tensor quantization
    {
        3ul,
        -1,
        {256ul, ov::Shape{1, 1, 1, 1, 1}, {-12.8f}, {12.7f}, {-12.8f}, {12.7f}},
        {255ul, ov::Shape { 1, 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f }},
        true,
        "Convolution",
        "i8"
    },
};

const std::vector<bool> addPrecisionPreserved = {false};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, GroupConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(params),
        ::testing::ValuesIn(addPrecisionPreserved)),
    GroupConvolutionTransformation::getTestCaseName);
}  // namespace i8_3d
}  // namespace

