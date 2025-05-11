// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <gtest/gtest.h>

#include "low_precision_transformations/pad_transformation.hpp"


using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<ov::PartialShape> inputShapes = {
    { 1, 3, 16, 16},
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

namespace commonTestCases {

const std::vector<ov::op::PadMode> padModes = {
    ov::op::PadMode::CONSTANT,
    ov::op::PadMode::EDGE,
    ov::op::PadMode::REFLECT,
    ov::op::PadMode::SYMMETRIC
};

const std::vector<LayerTestsDefinitions::PadTransformationParam> params = {
    // tensor quantization
    {
        { 256ul, ov::Shape{ 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 12.8f } },
        { 0, 0, 1, 1 },
        { 0, 0, 1, 1 },
    },
    // per-channel quantization
    {
        {
            256ul,
            ov::Shape{ 1, 3, 1, 1 },
            { -127.f, 0.f, 128.f / 2.f },
            { 128.f / 4.f, 128.f / 2.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f / 4.f, 255.f / 2.f, 255.f }
        },
        { 0, 0, 1, 1 },
        { 0, 0, 1, 1 },
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(padModes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    PadTransformation::getTestCaseName);
} // namespace commonTestCases

namespace testCasesForConstantMode {

const std::vector<LayerTestsDefinitions::PadTransformationParam> params = {
    // tensor quantization
    {
        { 256ul, ov::Shape{ 1, 1, 1, 1 }, { -2.f }, { 10.5f }, { -2.f }, { 10.5f } },
        { 0, 0, 1, 1 },
        { 0, 0, 1, 1 },
    },
    {
            { 256ul, ov::Shape{ 1, 1, 1, 1 }, { -2.f }, { 10.5f }, { -2.f }, { 10.5f } },
            { 0, 0, -1, 1 },
            { 0, 0, 1, -1 },
    },
    {
            { 256ul, ov::Shape{ 1, 1, 1, 1 }, { -2.f }, { 10.5f }, { -2.f }, { 10.5f } },
            { 0, 0, -1, -1 },
            { 0, 0, -1, -1 },
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(ov::op::PadMode::CONSTANT),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForConstantMode

namespace testCasesForOtherModes {

const std::vector<ov::op::PadMode> modesWithoutConstant = {
    ov::op::PadMode::EDGE,
    ov::op::PadMode::REFLECT,
    ov::op::PadMode::SYMMETRIC
};

const std::vector<LayerTestsDefinitions::PadTransformationParam> params = {
    // tensor quantization
    {
        { 256ul, ov::Shape{ 1, 1, 1, 1 }, { -2.f }, { 10.5f }, { -2.f }, { 10.5f } },
        { 0, 0, 1, 1 },
        { 0, 0, 1, 1 },
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(modesWithoutConstant),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForOtherModes

}  // namespace
