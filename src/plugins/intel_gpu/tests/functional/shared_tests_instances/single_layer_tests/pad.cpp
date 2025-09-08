// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "single_op_tests/pad.hpp"

namespace {
using ov::test::PadLayerTest;
using ov::test::Pad12LayerTest;
using ov::op::PadMode;

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16
};

const std::vector<std::vector<int64_t>> padsBegin2D = {{0, 0}, {1, 1}, {2, 0}, {0, 3}};
const std::vector<std::vector<int64_t>> padsEnd2D   = {{0, 0}, {1, 1}, {0, 1}, {3, 2}};
const std::vector<float> argPadValue = {0.f, 1.f, 2.f, -1.f};

const std::vector<PadMode> padMode = {
        PadMode::EDGE,
        PadMode::REFLECT,
        PadMode::SYMMETRIC
};

INSTANTIATE_TEST_SUITE_P(smoke_Pad2DConst,
                         PadLayerTest,
                         testing::Combine(testing::ValuesIn(padsBegin2D),
                                          testing::ValuesIn(padsEnd2D),
                                          testing::ValuesIn(argPadValue),
                                          testing::Values(PadMode::CONSTANT),
                                          testing::ValuesIn(netPrecisions),
                                          testing::Values(ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>{{13, 5}})),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         PadLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Pad2D,
                         PadLayerTest,
                         testing::Combine(testing::ValuesIn(padsBegin2D),
                                          testing::ValuesIn(padsEnd2D),
                                          testing::Values(0),
                                          testing::ValuesIn(padMode),
                                          testing::ValuesIn(netPrecisions),
                                          testing::Values(ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>{{13, 5}})),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         PadLayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> padsBegin4D = {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 0, 1, 0}, {0, 3, 0, 1}};
const std::vector<std::vector<int64_t>> padsEnd4D   = {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 0, 0, 1}, {1, 3, 2, 0}};

INSTANTIATE_TEST_SUITE_P(smoke_Pad4DConst,
                         PadLayerTest,
                         testing::Combine(testing::ValuesIn(padsBegin4D),
                                          testing::ValuesIn(padsEnd4D),
                                          testing::ValuesIn(argPadValue),
                                          testing::Values(PadMode::CONSTANT),
                                          testing::ValuesIn(netPrecisions),
                                          testing::Values(ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>{{3, 5, 10, 11}})),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         PadLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Pad4D,
                         PadLayerTest,
                         testing::Combine(testing::ValuesIn(padsBegin4D),
                                          testing::ValuesIn(padsEnd4D),
                                          testing::Values(0.0f),
                                          testing::ValuesIn(padMode),
                                          testing::ValuesIn(netPrecisions),
                                          testing::Values(ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>{{3, 5, 10, 11}})),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         PadLayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> padsBegin2DMixed = {{0, 0}, {1, 1}, {-2, 0}, {0, 3}, {2, -2}};
const std::vector<std::vector<int64_t>> padsEnd2DMixed   = {{0, 0}, {1, 1}, {0, 1}, {-3, -2}, {2, -1}};

INSTANTIATE_TEST_SUITE_P(smoke_Pad2DConst,
                         Pad12LayerTest,
                         testing::Combine(testing::ValuesIn(padsEnd2DMixed),
                                          testing::ValuesIn(padsEnd2D),
                                          testing::ValuesIn(argPadValue),
                                          testing::Values(PadMode::CONSTANT),
                                          testing::ValuesIn(netPrecisions),
                                          testing::Values(ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>{{13, 5}})),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         PadLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Pad2D,
                         Pad12LayerTest,
                         testing::Combine(testing::ValuesIn(padsBegin2DMixed),
                                          testing::ValuesIn(padsEnd2DMixed),
                                          testing::Values(-333.0f),
                                          testing::ValuesIn(padMode),
                                          testing::ValuesIn(netPrecisions),
                                          testing::Values(ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>{{13, 5}})),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         PadLayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> padsBegin4DMixed = {{0, 0, 0, 0}, {0, 3, 0, 0}, {0, 0, 0, 1}, {0, 0, -1, 1}, {2, 0, 0, 0}, {0, 3, 0, -1}};
const std::vector<std::vector<int64_t>> padsEnd4DMixed   = {{0, 0, 0, 0}, {0, 3, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 2}, {1, -3, 0, 0}, {0, 3, 0, -1}};

INSTANTIATE_TEST_SUITE_P(smoke_Pad4DConst,
                         Pad12LayerTest,
                         testing::Combine(testing::ValuesIn(padsBegin4DMixed),
                                          testing::ValuesIn(padsEnd4DMixed),
                                          testing::ValuesIn(argPadValue),
                                          testing::Values(PadMode::CONSTANT),
                                          testing::ValuesIn(netPrecisions),
                                          testing::Values(ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>{{3, 5, 10, 11}})),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         PadLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Pad4D,
                         Pad12LayerTest,
                         testing::Combine(testing::ValuesIn(padsBegin4DMixed),
                                          testing::ValuesIn(padsEnd4DMixed),
                                          testing::Values(-333),
                                          testing::ValuesIn(padMode),
                                          testing::ValuesIn(netPrecisions),
                                          testing::Values(ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>{{3, 5, 10, 11}})),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         PadLayerTest::getTestCaseName);

}  // namespace
