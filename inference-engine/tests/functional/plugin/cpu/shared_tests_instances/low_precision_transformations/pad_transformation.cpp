// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <gtest/gtest.h>

#include "low_precision_transformations/pad_transformation.hpp"


using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<ngraph::Shape> inputShapes = {
    { 1, 3, 16, 16},
    { 4, 3, 16, 16}
};

const std::vector<ngraph::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

namespace commonTestCases {

const std::vector<ngraph::op::PadMode> padModes = {
    ngraph::op::PadMode::CONSTANT,
    ngraph::op::PadMode::EDGE,
    ngraph::op::PadMode::REFLECT,
    ngraph::op::PadMode::SYMMETRIC
};

const std::vector<LayerTestsDefinitions::PadTransformationParam> params = {
    // tensor quantization
    {
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 12.8f } },
        { 0, 0, 1, 1 },
        { 0, 0, 1, 1 },
        "Pad",
        "U8"
    },
    // per-channel quantization with the same values
    {
        {
            256ul, ngraph::Shape{ 1, 3, 1, 1 },
            { -127.f, -127.f, -127.f },
            { 128.f, 128.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f, 255.f }
        },
        { 0, 0, 1, 1 },
        { 0, 0, 1, 1 },
        "Pad",
        "U8"
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
        { 0, 0, 1, 1 },
        { 0, 0, 1, 1 },
        "Pad",
        "U8"
    }
};

INSTANTIATE_TEST_CASE_P(smoke_LPT, PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(padModes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    PadTransformation::getTestCaseName);
} // namespace commonTestCases

namespace testCasesForConstantMode {

const std::vector<LayerTestsDefinitions::PadTransformationParam> params = {
    // tensor quantization
    {
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { -2.f }, { 10.5f }, { -2.f }, { 10.5f } },
        { 0, 0, 1, 1 },
        { 0, 0, 1, 1 },
        "Pad",
        "FP32"
    },
};

INSTANTIATE_TEST_CASE_P(smoke_LPT, PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(ngraph::op::PadMode::CONSTANT),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForConstantMode

namespace testCasesForOtherModes {

const std::vector<ngraph::op::PadMode> modesWithoutConstant = {
    ngraph::op::PadMode::EDGE,
    ngraph::op::PadMode::REFLECT,
    ngraph::op::PadMode::SYMMETRIC
};

const std::vector<LayerTestsDefinitions::PadTransformationParam> params = {
    // tensor quantization
    {
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { -2.f }, { 10.5f }, { -2.f }, { 10.5f } },
        { 0, 0, 1, 1 },
        { 0, 0, 1, 1 },
        "Pad",
        "U8"
    },
};

INSTANTIATE_TEST_CASE_P(smoke_LPT, PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(modesWithoutConstant),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForOtherModes

}  // namespace
