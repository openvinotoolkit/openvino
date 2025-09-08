// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/broadcast.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::BroadcastLayerTest;
const std::vector<ov::element::Type> inputPrecisions = {
        ov::element::f32,
        ov::element::bf16,
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
        {{ 1 }}
};

const auto numpyBroadcast1DInputParams = ::testing::Combine(
        ::testing::ValuesIn(targetShapesNumpy1D),
        ::testing::Values(ov::AxisSet{}), //not used in numpy mode
        ::testing::Values(ov::op::BroadcastType::NUMPY),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_1d_static)),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestNumpyBroadcast1D, BroadcastLayerTest, numpyBroadcast1DInputParams, BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PrecTransformation, BroadcastLayerTest,
        ::testing::Combine(
            ::testing::Values(targetShapesNumpy1D[0]),
            ::testing::Values(ov::AxisSet{}), //not used in numpy mode
            ::testing::Values(ov::op::BroadcastType::NUMPY),
            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_1d_static)),
            ::testing::ValuesIn(inputTPrecisions),
            ::testing::Values(ov::test::utils::DEVICE_CPU)),
        BroadcastLayerTest::getTestCaseName);

// 2D
std::vector<std::vector<size_t>> targetShapesNumpy2D = {
        {3, 1},
        {3, 6},
        {2, 3, 6},
        {2, 2, 3, 6},
};

const std::vector<std::vector<ov::Shape>> input_shapes_2d_static = {
        {{ 3, 1 }}
};

const auto numpyBroadcast2DInputParams = ::testing::Combine(
        ::testing::ValuesIn(targetShapesNumpy2D),
        ::testing::Values(ov::AxisSet{}), //not used in numpy mode
        ::testing::Values(ov::op::BroadcastType::NUMPY),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_2d_static)),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestNumpyBroadcast2D, BroadcastLayerTest, numpyBroadcast2DInputParams, BroadcastLayerTest::getTestCaseName);

// 3D
std::vector<std::vector<size_t>> targetShapesNumpy3D = {
        {1, 4, 1},
        {1, 4, 4},
        {1, 1, 4, 4},
        {2, 1, 1, 4, 4},
};

const std::vector<std::vector<ov::Shape>> input_shapes_3d_static = {
        {{ 1, 4, 1 }}
};

const auto numpyBroadcast3DInputParams = ::testing::Combine(
        ::testing::ValuesIn(targetShapesNumpy3D),
        ::testing::Values(ov::AxisSet{}), //not used in numpy mode
        ::testing::Values(ov::op::BroadcastType::NUMPY),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_3d_static)),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestNumpyBroadcast3D, BroadcastLayerTest, numpyBroadcast3DInputParams, BroadcastLayerTest::getTestCaseName);

const std::vector<std::vector<ov::Shape>> evaluate_shapes_static = {
        {{ 1, 2, 1, 4, 1, 6, 1, 8, 1, 10 }}
};

// EVALUATE
const auto numpyBroadcastEvaluateParams = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
        ::testing::Values(ov::AxisSet{}), //not used in numpy mode
        ::testing::Values(ov::op::BroadcastType::NUMPY),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(evaluate_shapes_static)),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestNumpyBroadcastEvaluate,
                        BroadcastLayerTest,
                        numpyBroadcastEvaluateParams,
                        BroadcastLayerTest::getTestCaseName);
// END NUMPY MODE //////////////////////////////////////

// BIDIRECTIONAL MODE //////////////////////////////////
std::vector<std::vector<ov::Shape>> shapes_bidi_static = {
        {{4, 1}},
        {{1, 4, 1}},
        {{4, 1, 1}}
};

std::vector<std::vector<size_t>> targetShapesBidi = {
        {4, 1, 4},
        {1, 4, 4},
        {1, 1, 4, 4}
};

const auto bidirectionalBroadcastParams = ::testing::Combine(
        ::testing::ValuesIn(targetShapesBidi),
        ::testing::Values(ov::AxisSet{}), //not used in bidirectional mode
        ::testing::Values(ov::op::BroadcastType::BIDIRECTIONAL),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_bidi_static)),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestBidirectionalBroadcast, BroadcastLayerTest, bidirectionalBroadcastParams, BroadcastLayerTest::getTestCaseName);

// EXPLICIT MODE ///////////////////////////////////////
// 1D

std::vector<std::vector<ov::Shape>> input_shapes_explicit_1d_static = {
        {{ 4 }}
};
std::vector<std::vector<size_t>> targetShapesExplicit1D = { {4, 2, 4}, {4, 2, 4, 1} };
std::vector<ov::AxisSet> axes1D = { {0}, {2} };

const auto explicitBroadcast1DInputParams = ::testing::Combine(
        ::testing::ValuesIn(targetShapesExplicit1D),
        ::testing::ValuesIn(axes1D),
        ::testing::Values(ov::op::BroadcastType::EXPLICIT),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_explicit_1d_static)),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestExplicitBroadcast1D, BroadcastLayerTest, explicitBroadcast1DInputParams, BroadcastLayerTest::getTestCaseName);

const auto bidirectionalBroadcastParams3 = ::testing::Combine(
        ::testing::Values(targetShapesBidi[2]),
        ::testing::Values(ov::AxisSet{}), //not used in bidirectional mode
        ::testing::Values(ov::op::BroadcastType::BIDIRECTIONAL),
        ::testing::Values(ov::test::static_shapes_to_test_representation(shapes_bidi_static[2])),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_TestBidirectionalBroadcast3,
        BroadcastLayerTest,
        bidirectionalBroadcastParams3,
        BroadcastLayerTest::getTestCaseName
);

// EXPLICIT MODE

std::vector<std::vector<size_t>> inShapesExplicit = {
        {3, 1},
        {2, 4}
};

std::vector<std::vector<size_t>> targetShapesExplicit = {
        {2, 3, 1},
        {2, 3, 4}
};

// 2D
std::vector<std::vector<ov::Shape>> input_shapes_explicit_2d_static = {
        {{ 2, 4 }}
};

std::vector<std::vector<size_t>> targetShapesExplicit2D = { {2, 2, 4}, {2, 2, 4, 1}};
std::vector<ov::AxisSet> axes2D = { {1, 2}, {0, 2} };

const auto explicitBroadcast2DInputParams = ::testing::Combine(
        ::testing::ValuesIn(targetShapesExplicit2D),
        ::testing::ValuesIn(axes2D),
        ::testing::Values(ov::op::BroadcastType::EXPLICIT),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_explicit_2d_static)),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestExplicitBroadcast2D, BroadcastLayerTest, explicitBroadcast2DInputParams, BroadcastLayerTest::getTestCaseName);

// 3D
std::vector<std::vector<ov::Shape>> input_shapes_explicit_3d_static = {
        {{ 2, 2, 2 }}
};
std::vector<std::vector<size_t>> targetShapesExplicit3D = { {2, 2, 2, 2} };
std::vector<ov::AxisSet> axes3D = { {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3} };

const auto explicitBroadcast3DInputParams = ::testing::Combine(
        ::testing::ValuesIn(targetShapesExplicit3D),
        ::testing::ValuesIn(axes3D),
        ::testing::Values(ov::op::BroadcastType::EXPLICIT),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_explicit_3d_static)),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestExplicitBroadcast3D, BroadcastLayerTest, explicitBroadcast3DInputParams, BroadcastLayerTest::getTestCaseName);
// END EXPLICIT MODE ///////////////////////////////////

}  // namespace

