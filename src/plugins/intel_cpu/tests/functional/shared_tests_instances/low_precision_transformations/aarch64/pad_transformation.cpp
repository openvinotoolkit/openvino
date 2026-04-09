// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <gtest/gtest.h>

#include "low_precision_transformations/pad_transformation.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32
};

const std::vector<ov::PartialShape> inputShapes = {
    {1, 3, 16, 16},
    {4, 3, 16, 16}
};

namespace commonTestCases {
const std::vector<ov::op::PadMode> padModes = {
    ov::op::PadMode::CONSTANT,
    ov::op::PadMode::EDGE,
    ov::op::PadMode::REFLECT,
    ov::op::PadMode::SYMMETRIC
};

const std::vector<LayerTestsDefinitions::PadTransformationParam> params = {
    {{256ul, ov::Shape{1, 1, 1, 1}, {0.f}, {25.5f}, {0.f}, {12.8f}}, {0, 0, 1, 1}, {0, 0, 1, 1}, 0.f, "Pad", "u8"},
    {{{256ul, ov::Shape{1, 3, 1, 1}, {-127.f, -127.f, -127.f}, {128.f, 128.f, 128.f}, {0.f, 0.f, 0.f}, {255.f, 255.f, 255.f}}},
     {0, 0, 1, 1},
     {0, 0, 1, 1},
     0.f,
     "Pad",
     "u8"},
    {{{256ul, ov::Shape{1, 3, 1, 1}, {-127.f, 0.f, 128.f / 2.f}, {128.f / 4.f, 128.f / 2.f, 128.f}, {0.f, 0.f, 0.f}, {255.f / 4.f, 255.f / 2.f, 255.f}}},
     {0, 0, 1, 1},
     {0, 0, 1, 1},
     0.f,
     "Pad",
     "u8"},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         PadTransformation,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(padModes),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::ValuesIn(params)),
                         PadTransformation::getTestCaseName);
}  // namespace commonTestCases
}  // namespace