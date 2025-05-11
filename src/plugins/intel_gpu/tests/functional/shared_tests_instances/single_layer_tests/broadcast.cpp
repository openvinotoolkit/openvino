// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/broadcast.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::BroadcastLayerTest;
using ov::test::BroadcastParamsTuple;
const std::vector<ov::element::Type> inputPrecisions = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
        ov::element::i8,
        ov::element::u8
};

const std::vector<ov::element::Type> inputTPrecisions = {
        ov::element::f16,
        ov::element::i16,
        ov::element::boolean
};

// NUMPY MODE //////////////////////////////////////////
// 0D
std::vector<std::vector<size_t>> targetShapesNumpy0D = {
        {},
};

std::vector<std::vector<ov::Shape>> input_shapes_0d_static = {
        {{}}
};

INSTANTIATE_TEST_SUITE_P(smoke_TestNumpyBroadcast0D,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::ValuesIn(targetShapesNumpy0D),
                                           ::testing::Values(ov::AxisSet{}),  // not used in numpy mode
                                           ::testing::Values(ov::op::BroadcastType::NUMPY),
                                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_0d_static)),
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        BroadcastLayerTest::getTestCaseName);

// NUMPY MODE //////////////////////////////////////////
// 1D
std::vector<std::vector<size_t>> targetShapesNumpy1D = {
        {1},
        {16},
        {1, 16},
        {1, 1, 16},
        {2, 16},
        {2, 3, 6},
        {1, 4, 4},
};

std::vector<std::vector<ov::Shape>> input_shapes_1d_static = {
        {{1}}
};

INSTANTIATE_TEST_SUITE_P(smoke_TestNumpyBroadcast1D,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::ValuesIn(targetShapesNumpy1D),
                                           ::testing::Values(ov::AxisSet{}),  // not used in numpy mode
                                           ::testing::Values(ov::op::BroadcastType::NUMPY),
                                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_1d_static)),
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PrecTransformation, BroadcastLayerTest,
        ::testing::Combine(
            ::testing::Values(targetShapesNumpy1D[0]),
            ::testing::Values(ov::AxisSet{}), //not used in numpy mode
            ::testing::Values(ov::op::BroadcastType::NUMPY),
            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_1d_static)),
            ::testing::ValuesIn(inputTPrecisions),
            ::testing::Values(ov::test::utils::DEVICE_GPU)),
        BroadcastLayerTest::getTestCaseName);

// 2D
std::vector<std::vector<size_t>> targetShapesNumpy2D = {
        {3, 1},
        {3, 6},
        {2, 3, 6},
        {2, 2, 3, 6},
};

std::vector<std::vector<ov::Shape>> input_shapes_2d_static = {
        {{3, 1}}
};

INSTANTIATE_TEST_SUITE_P(smoke_TestNumpyBroadcast2D,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::ValuesIn(targetShapesNumpy2D),
                                           ::testing::Values(ov::AxisSet{}),  // not used in numpy mode
                                           ::testing::Values(ov::op::BroadcastType::NUMPY),
                                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_2d_static)),
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        BroadcastLayerTest::getTestCaseName);

// 3D
std::vector<std::vector<size_t>> targetShapesNumpy3D = {
        {1, 4, 1},
        {1, 4, 4},
        {1, 1, 4, 4},
        {2, 1, 1, 4, 4},
};

std::vector<std::vector<ov::Shape>> input_shapes_3d_static = {
        {{1, 4, 1}}
};


INSTANTIATE_TEST_SUITE_P(smoke_TestNumpyBroadcast3D,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::ValuesIn(targetShapesNumpy3D),
                                           ::testing::Values(ov::AxisSet{}),  // not used in numpy mode
                                           ::testing::Values(ov::op::BroadcastType::NUMPY),
                                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_3d_static)),
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        BroadcastLayerTest::getTestCaseName);

std::vector<std::vector<size_t>> targetShapesNumpy6D = {
        {1, 2, 3, 4, 5, 6},
};

std::vector<std::vector<ov::Shape>> input_shapes_6d_static = {
        {{1, 2, 1, 4, 1, 6}}
};

INSTANTIATE_TEST_SUITE_P(smoke_TestNumpyBroadcast6D,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::ValuesIn(targetShapesNumpy6D),
                                           ::testing::Values(ov::AxisSet{}),  // not used in numpy mode
                                           ::testing::Values(ov::op::BroadcastType::NUMPY),
                                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_6d_static)),
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        BroadcastLayerTest::getTestCaseName);

std::vector<std::vector<size_t>> targetShapesNumpy5D = {
        {1, 2, 3, 4, 5},
};

std::vector<std::vector<ov::Shape>> input_shapes_5d_static = {
        {{1, 2, 1, 4, 1}}
};

INSTANTIATE_TEST_SUITE_P(smoke_TestNumpyBroadcast5D,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::ValuesIn(targetShapesNumpy5D),
                                           ::testing::Values(ov::AxisSet{}),  // not used in numpy mode
                                           ::testing::Values(ov::op::BroadcastType::NUMPY),
                                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_5d_static)),
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        BroadcastLayerTest::getTestCaseName);
// END NUMPY MODE //////////////////////////////////////

// BIDIRECTIONAL MODE //////////////////////////////////
std::vector<std::vector<ov::Shape>> inShapesBidi = {
        {{4, 1}},
        {{1, 4, 1}},
        {{4, 1, 1}}
};

std::vector<std::vector<size_t>> targetShapesBidi = {
        {4, 1, 4},
        {1, 4, 4},
        {1, 1, 4, 4}
};

INSTANTIATE_TEST_SUITE_P(smoke_TestBidirectionalBroadcast,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::ValuesIn(targetShapesBidi),
                                           ::testing::Values(ov::AxisSet{}),  // not used in bidirectional mode
                                           ::testing::Values(ov::op::BroadcastType::BIDIRECTIONAL),
                                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesBidi)),
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        BroadcastLayerTest::getTestCaseName);

// EXPLICIT MODE ///////////////////////////////////////
// 1D
std::vector<std::vector<ov::Shape>> inShapesExplicit1D = { {{4}} };
std::vector<std::vector<size_t>> targetShapesExplicit1D = { {4, 2, 4}, {4, 2, 4, 1} };
std::vector<ov::AxisSet> axes1D = { {0}, {2} };

INSTANTIATE_TEST_SUITE_P(smoke_TestExplicitBroadcast1D,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::ValuesIn(targetShapesExplicit1D),
                                           ::testing::ValuesIn(axes1D),
                                           ::testing::Values(ov::op::BroadcastType::EXPLICIT),
                                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesExplicit1D)),
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestBidirectionalBroadcast3,
                         BroadcastLayerTest,
                         ::testing::Combine(::testing::Values(targetShapesBidi[2]),
                                            ::testing::Values(ov::AxisSet{}),  // not used in bidirectional mode
                                            ::testing::Values(ov::op::BroadcastType::BIDIRECTIONAL),
                                            ::testing::Values(ov::test::static_shapes_to_test_representation(inShapesBidi[2])),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         BroadcastLayerTest::getTestCaseName);

// EXPLICIT MODE

std::vector<std::vector<ov::Shape>> inShapesExplicit = {
        {{3, 1}},
        {{2, 4}}
};

std::vector<std::vector<size_t>> targetShapesExplicit = {
        {2, 3, 1},
        {2, 3, 4}
};

// 2D
std::vector<std::vector<ov::Shape>> inShapesExplicit2D = { {{2, 4}} };
std::vector<std::vector<size_t>> targetShapesExplicit2D = { {2, 2, 4}, {2, 2, 4, 1}};
std::vector<ov::AxisSet> axes2D = { {1, 2}, {0, 2} };

INSTANTIATE_TEST_SUITE_P(smoke_TestExplicitBroadcast2D,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::ValuesIn(targetShapesExplicit2D),
                                           ::testing::ValuesIn(axes2D),
                                           ::testing::Values(ov::op::BroadcastType::EXPLICIT),
                                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesExplicit2D)),
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        BroadcastLayerTest::getTestCaseName);

// 3D
std::vector<std::vector<ov::Shape>> inShapesExplicit3D = { {{2, 2, 2}} };
std::vector<std::vector<size_t>> targetShapesExplicit3D = { {2, 2, 2, 2} };
std::vector<ov::AxisSet> axes3D = { {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3} };

INSTANTIATE_TEST_SUITE_P(smoke_TestExplicitBroadcast3D,
                        BroadcastLayerTest,
                        ::testing::Combine(::testing::ValuesIn(targetShapesExplicit3D),
                                           ::testing::ValuesIn(axes3D),
                                           ::testing::Values(ov::op::BroadcastType::EXPLICIT),
                                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesExplicit3D)),
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        BroadcastLayerTest::getTestCaseName);
// END EXPLICIT MODE ///////////////////////////////////
}  // namespace
