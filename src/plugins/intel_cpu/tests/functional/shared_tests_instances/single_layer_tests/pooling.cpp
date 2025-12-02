// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/pooling.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::AvgPoolingV16LayerTest;
using ov::test::avgPoolV16LayerTestParams;
using ov::test::MaxPoolingV8LayerTest;
using ov::test::PoolingLayerTest;
using ov::test::poolSpecificParams;
using ov::test::utils::PoolingTypes;

const std::vector<ov::element::Type> model_types = {ov::element::f16};

const std::vector<std::vector<size_t>> kernels = {{3, 3}, {3, 5}};
const std::vector<std::vector<size_t>> kernel_3d = {{2, 2, 2}};

const std::vector<ov::Strides> strides = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};

const std::vector<ov::Strides> strides_3d = {{1, 1, 1}, {2, 2, 2}};

const std::vector<std::vector<size_t>> pad_begins = {{0, 0}, {0, 2}};

const std::vector<std::vector<size_t>> pad_begins_3d = {{0, 0, 0}};

const std::vector<std::vector<size_t>> pad_ends = {{0, 0}, {0, 2}};

const std::vector<std::vector<size_t>> pad_ends_3d = {{0, 0, 0}};

////* ========== Max Polling ========== */
/* +========== Explicit Pad Floor Rounding ========== */
std::vector<ov::Shape> input_shapes_static = {{1, 3, 30, 30}};

const auto maxPool_ExplicitPad_FloorRounding_Params =
    ::testing::Combine(::testing::Values(PoolingTypes::MAX),
                       ::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::ValuesIn(pad_begins),
                       ::testing::ValuesIn(pad_ends),
                       ::testing::Values(ov::op::RoundingType::FLOOR),
                       ::testing::Values(ov::op::PadType::EXPLICIT),
                       ::testing::Values(false));  // placeholder value - exclude pad not applicable for max pooling

INSTANTIATE_TEST_SUITE_P(
    smoke_MaxPool_ExplicitPad_FloorRounding,
    PoolingLayerTest,
    ::testing::Combine(maxPool_ExplicitPad_FloorRounding_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    PoolingLayerTest::getTestCaseName);

/* +========== Same Upper Pad Floor Rounding ========== */
const auto maxPool_SameUpperPad_FloorRounding_Params =
    ::testing::Combine(::testing::Values(PoolingTypes::MAX),
                       ::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::ValuesIn(pad_begins),
                       ::testing::ValuesIn(pad_ends),
                       ::testing::Values(ov::op::RoundingType::FLOOR),
                       ::testing::Values(ov::op::PadType::SAME_UPPER),
                       ::testing::Values(false));  // placeholder value - exclude pad not applicable for max pooling

INSTANTIATE_TEST_SUITE_P(
    smoke_MaxPool_SameUpperPad_FloorRounding,
    PoolingLayerTest,
    ::testing::Combine(maxPool_SameUpperPad_FloorRounding_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    PoolingLayerTest::getTestCaseName);

/* +========== Same Lower Pad Floor Rounding ========== */
const auto maxPool_SameLowerPad_FloorRounding_Params =
    ::testing::Combine(::testing::Values(PoolingTypes::MAX),
                       ::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::ValuesIn(pad_begins),
                       ::testing::ValuesIn(pad_ends),
                       ::testing::Values(ov::op::RoundingType::FLOOR),
                       ::testing::Values(ov::op::PadType::SAME_LOWER),
                       ::testing::Values(false));  // placeholder value - exclude pad not applicable for max pooling

INSTANTIATE_TEST_SUITE_P(
    smoke_MaxPool_SameLowerPad_FloorRounding,
    PoolingLayerTest,
    ::testing::Combine(maxPool_SameUpperPad_FloorRounding_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    PoolingLayerTest::getTestCaseName);

/* ========== Explicit Pad Floor Rounding 5D input========== */

std::vector<ov::Shape> input_shapes_5d_static = {{32, 32, 2, 2, 2}};

const auto maxPool_ExplicitPad_FloorRounding_5Dinput_Params =
    ::testing::Combine(::testing::Values(PoolingTypes::MAX),
                       ::testing::ValuesIn(kernel_3d),
                       ::testing::ValuesIn(strides_3d),
                       ::testing::ValuesIn(pad_begins_3d),
                       ::testing::ValuesIn(pad_ends_3d),
                       ::testing::Values(ov::op::RoundingType::FLOOR),
                       ::testing::Values(ov::op::PadType::EXPLICIT),
                       ::testing::Values(false));  // placeholder value - exclude pad not applicable for max pooling

INSTANTIATE_TEST_SUITE_P(
    smoke_MaxPool_ExplicitPad_FloorRounding_5Dinput,
    PoolingLayerTest,
    ::testing::Combine(maxPool_ExplicitPad_FloorRounding_5Dinput_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_5d_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    PoolingLayerTest::getTestCaseName);

/* ========== Same Upper Pad Floor Rounding 5D input========== */
const auto maxPool_SameUpperPad_FloorRounding_5Dinput_Params =
    ::testing::Combine(::testing::Values(PoolingTypes::MAX),
                       ::testing::ValuesIn(kernel_3d),
                       ::testing::ValuesIn(strides_3d),
                       ::testing::ValuesIn(pad_begins_3d),
                       ::testing::ValuesIn(pad_ends_3d),
                       ::testing::Values(ov::op::RoundingType::FLOOR),
                       ::testing::Values(ov::op::PadType::SAME_UPPER),
                       ::testing::Values(false));  // placeholder value - exclude pad not applicable for max pooling

INSTANTIATE_TEST_SUITE_P(
    smoke_MaxPool_SameUpperPad_FloorRounding_5Dinput,
    PoolingLayerTest,
    ::testing::Combine(maxPool_SameUpperPad_FloorRounding_5Dinput_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_5d_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    PoolingLayerTest::getTestCaseName);

/* ========== Same Lower Pad Ceil Rounding 5D input========== */
const auto maxPool_SameLowerPad_CeilRounding_5Dinput_Params =
    ::testing::Combine(::testing::Values(PoolingTypes::MAX),
                       ::testing::ValuesIn(kernel_3d),
                       ::testing::ValuesIn(strides_3d),
                       ::testing::ValuesIn(pad_begins_3d),
                       ::testing::ValuesIn(pad_ends_3d),
                       ::testing::Values(ov::op::RoundingType::CEIL),
                       ::testing::Values(ov::op::PadType::SAME_LOWER),
                       ::testing::Values(false));  // placeholder value - exclude pad not applicable for max pooling

INSTANTIATE_TEST_SUITE_P(
    smoke_MaxPool_SameLowerPad_CeilRounding_5Dinput,
    PoolingLayerTest,
    ::testing::Combine(maxPool_SameUpperPad_FloorRounding_5Dinput_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_5d_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    PoolingLayerTest::getTestCaseName);

/* ========== Explicit Pad Ceil Rounding ========== */
const auto maxPool_ExplicitPad_CeilRounding_Params =
    ::testing::Combine(::testing::Values(PoolingTypes::MAX),
                       ::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::ValuesIn(pad_begins),
                       ::testing::ValuesIn(pad_ends),
                       ::testing::Values(ov::op::RoundingType::CEIL),
                       ::testing::Values(ov::op::PadType::EXPLICIT),
                       ::testing::Values(false));  // placeholder value - exclude pad not applicable for max pooling

INSTANTIATE_TEST_SUITE_P(
    smoke_MaxPool_ExplicitPad_CeilRounding,
    PoolingLayerTest,
    ::testing::Combine(maxPool_ExplicitPad_CeilRounding_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    PoolingLayerTest::getTestCaseName);

////* ========== Avg Pooling ========== */
/* +========== Explicit Pad Ceil Rounding ========== */
const auto avgPoolExplicitPadCeilRoundingParams =
    ::testing::Combine(::testing::Values(PoolingTypes::AVG),
                       ::testing::ValuesIn(kernels),
                       // TODO: Non 1 strides fails in reference implementation with error "The end corner is out of
                       // bounds at axis 3" thrown in the test body.
                       ::testing::ValuesIn(strides),
                       ::testing::ValuesIn(std::vector<std::vector<size_t>>({{0, 0}, {1, 1}, {0, 1}})),
                       ::testing::ValuesIn(std::vector<std::vector<size_t>>({{0, 0}, {1, 1}, {0, 1}})),
                       ::testing::Values(ov::op::RoundingType::CEIL),
                       ::testing::Values(ov::op::PadType::EXPLICIT),
                       ::testing::Values(true, false));

INSTANTIATE_TEST_SUITE_P(
    smoke_AvgPool_ExplicitPad_CeilRounding,
    PoolingLayerTest,
    ::testing::Combine(avgPoolExplicitPadCeilRoundingParams,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    PoolingLayerTest::getTestCaseName);

std::vector<poolSpecificParams> psParams({poolSpecificParams(PoolingTypes::AVG,
                                                             {2, 2},
                                                             {2, 2},
                                                             {0, 0},
                                                             {0, 0},
                                                             ov::op::RoundingType::CEIL,
                                                             ov::op::PadType::EXPLICIT,
                                                             false),
                                          poolSpecificParams(PoolingTypes::AVG,
                                                             {7, 7},
                                                             {1, 1},
                                                             {0, 0},
                                                             {1, 1},
                                                             ov::op::RoundingType::CEIL,
                                                             ov::op::PadType::EXPLICIT,
                                                             false)});

std::vector<ov::Shape> input_shapes_explicit_pad_ceil_rounding_corner_static = {{1, 3, 30, 30}};

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_ExplicitPad_CeilRounding_corner,
                         PoolingLayerTest,
                         ::testing::Combine(::testing::ValuesIn(psParams),
                                            ::testing::ValuesIn(model_types),
                                            ::testing::Values(ov::test::static_shapes_to_test_representation(
                                                input_shapes_explicit_pad_ceil_rounding_corner_static)),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         PoolingLayerTest::getTestCaseName);

/* +========== Explicit Pad Floor Rounding ========== */
const auto avgPoolExplicitPadFloorRoundingParams =
    ::testing::Combine(::testing::Values(PoolingTypes::AVG),
                       ::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::ValuesIn(std::vector<std::vector<size_t>>({{0, 0}, {1, 1}})),
                       ::testing::ValuesIn(std::vector<std::vector<size_t>>({{0, 0}, {1, 1}})),
                       ::testing::Values(ov::op::RoundingType::FLOOR),
                       ::testing::Values(ov::op::PadType::EXPLICIT),
                       ::testing::Values(true, false));

INSTANTIATE_TEST_SUITE_P(
    smoke_AvgPool_ExplicitPad_FloorRounding,
    PoolingLayerTest,
    ::testing::Combine(avgPoolExplicitPadFloorRoundingParams,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    PoolingLayerTest::getTestCaseName);

/* ========== Explicit Pad Floor Rounding 5D input========== */
const auto avgPool_ExplicitPad_FloorRounding_5Dinput_Params =
    ::testing::Combine(::testing::Values(PoolingTypes::AVG),
                       ::testing::ValuesIn(kernel_3d),
                       ::testing::ValuesIn(strides_3d),
                       ::testing::ValuesIn(pad_begins_3d),
                       ::testing::ValuesIn(pad_ends_3d),
                       ::testing::Values(ov::op::RoundingType::FLOOR),
                       ::testing::Values(ov::op::PadType::EXPLICIT),
                       ::testing::Values(true, false));

std::vector<ov::Shape> input_shapes_5d_2_static = {{32, 32, 2, 2, 4}};

INSTANTIATE_TEST_SUITE_P(
    smoke_AvgPool_ExplicitPad_FloorRounding_5Dinput,
    PoolingLayerTest,
    ::testing::Combine(avgPool_ExplicitPad_FloorRounding_5Dinput_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_5d_2_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    PoolingLayerTest::getTestCaseName);

/* ========== Same Upper Pad Floor Rounding 5D input========== */
const auto avgPool_SameUpperPad_FloorRounding_5Dinput_Params =
    ::testing::Combine(::testing::Values(PoolingTypes::AVG),
                       ::testing::ValuesIn(kernel_3d),
                       ::testing::ValuesIn(strides_3d),
                       ::testing::ValuesIn(pad_begins_3d),
                       ::testing::ValuesIn(pad_ends_3d),
                       ::testing::Values(ov::op::RoundingType::FLOOR),
                       ::testing::Values(ov::op::PadType::SAME_UPPER),
                       ::testing::Values(true));

INSTANTIATE_TEST_SUITE_P(
    smoke_AvgPool_SameUpperPad_FloorRounding_5Dinput,
    PoolingLayerTest,
    ::testing::Combine(avgPool_SameUpperPad_FloorRounding_5Dinput_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_5d_2_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    PoolingLayerTest::getTestCaseName);

/* ========== Same Lower Pad Ceil Rounding 5D input========== */
const auto avgPool_SameLowerPad_CeilRounding_5Dinput_Params =
    ::testing::Combine(::testing::Values(PoolingTypes::AVG),
                       ::testing::ValuesIn(kernel_3d),
                       ::testing::ValuesIn(strides_3d),
                       ::testing::ValuesIn(pad_begins_3d),
                       ::testing::ValuesIn(pad_ends_3d),
                       ::testing::Values(ov::op::RoundingType::CEIL),
                       ::testing::Values(ov::op::PadType::SAME_LOWER),
                       ::testing::Values(true));

INSTANTIATE_TEST_SUITE_P(
    smoke_AvgPool_SameLowerPad_CeilRounding_5Dinput,
    PoolingLayerTest,
    ::testing::Combine(avgPool_SameLowerPad_CeilRounding_5Dinput_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_5d_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    PoolingLayerTest::getTestCaseName);

////* ========== Max Pooling V8 ========== */

const std::vector<ov::Strides> dilation = {{1, 1}, {2, 2}};
const std::vector<ov::Strides> dilation_3d = {{1, 1, 1}, {2, 2, 2}};

/* ========== Explicit Pad Floor Rounding ========== */
const auto maxPoolv8_ExplicitPad_FloorRounding_Params =
    ::testing::Combine(::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::ValuesIn(dilation),
                       ::testing::ValuesIn(pad_begins),
                       ::testing::ValuesIn(pad_ends),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(0),
                       ::testing::Values(ov::op::RoundingType::FLOOR),
                       ::testing::Values(ov::op::PadType::EXPLICIT));

INSTANTIATE_TEST_SUITE_P(
    smoke_MaxPoolV8_ExplicitPad_FloorRounding,
    MaxPoolingV8LayerTest,
    ::testing::Combine(maxPoolv8_ExplicitPad_FloorRounding_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    MaxPoolingV8LayerTest::getTestCaseName);

/* ========== Same Upper Pad Floor Rounding ========== */
const auto maxPoolv8_SameUpperPad_FloorRounding_Params =
    ::testing::Combine(::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::ValuesIn(dilation),
                       ::testing::ValuesIn(pad_begins),
                       ::testing::ValuesIn(pad_ends),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(0),
                       ::testing::Values(ov::op::RoundingType::FLOOR),
                       ::testing::Values(ov::op::PadType::SAME_UPPER));

INSTANTIATE_TEST_SUITE_P(
    smoke_MaxPoolv8_SameUpperPad_FloorRounding,
    MaxPoolingV8LayerTest,
    ::testing::Combine(maxPoolv8_SameUpperPad_FloorRounding_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    MaxPoolingV8LayerTest::getTestCaseName);

/* ========== Same Lower Pad Floor Rounding ========== */
const auto maxPoolv8_SameLowerPad_FloorRounding_Params =
    ::testing::Combine(::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::ValuesIn(dilation),
                       ::testing::ValuesIn(pad_begins),
                       ::testing::ValuesIn(pad_ends),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(0),
                       ::testing::Values(ov::op::RoundingType::FLOOR),
                       ::testing::Values(ov::op::PadType::SAME_LOWER));

INSTANTIATE_TEST_SUITE_P(
    smoke_MaxPoolv8_SameLowerPad_FloorRounding,
    MaxPoolingV8LayerTest,
    ::testing::Combine(maxPoolv8_SameLowerPad_FloorRounding_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    MaxPoolingV8LayerTest::getTestCaseName);

/* ========= Explicit Pad Floor Rounding 5D input========== */
const auto maxPoolv8_ExplicitPad_FloorRounding_5Dinput_Params =
    ::testing::Combine(::testing::ValuesIn(kernel_3d),
                       ::testing::ValuesIn(strides_3d),
                       ::testing::Values(dilation_3d[0]),
                       ::testing::ValuesIn(pad_begins_3d),
                       ::testing::ValuesIn(pad_ends_3d),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(0),
                       ::testing::Values(ov::op::RoundingType::FLOOR),
                       ::testing::Values(ov::op::PadType::EXPLICIT));

INSTANTIATE_TEST_SUITE_P(
    smoke_MaxPoolv8_ExplicitPad_FloorRounding_5Dinput,
    MaxPoolingV8LayerTest,
    ::testing::Combine(maxPoolv8_ExplicitPad_FloorRounding_5Dinput_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_5d_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    MaxPoolingV8LayerTest::getTestCaseName);

/* ========= Same Upper Pad Floor Rounding 5D input========== */
const auto maxPoolv8_SameUpperPad_FloorRounding_5Dinput_Params =
    ::testing::Combine(::testing::ValuesIn(kernel_3d),
                       ::testing::ValuesIn(strides_3d),
                       ::testing::ValuesIn(dilation_3d),
                       ::testing::ValuesIn(pad_begins_3d),
                       ::testing::ValuesIn(pad_ends_3d),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(0),
                       ::testing::Values(ov::op::RoundingType::FLOOR),
                       ::testing::Values(ov::op::PadType::SAME_UPPER));

INSTANTIATE_TEST_SUITE_P(
    smoke_MaxPoolv8_SameUpperPad_FloorRounding_5Dinput,
    MaxPoolingV8LayerTest,
    ::testing::Combine(maxPoolv8_SameUpperPad_FloorRounding_5Dinput_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_5d_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    MaxPoolingV8LayerTest::getTestCaseName);

/* ========= Same Lower Pad Ceil Rounding 5D input========== */
const auto maxPoolv8_SameLowerPad_CeilRounding_5Dinput_Params =
    ::testing::Combine(::testing::ValuesIn(kernel_3d),
                       ::testing::ValuesIn(strides_3d),
                       ::testing::ValuesIn(dilation_3d),
                       ::testing::ValuesIn(pad_begins_3d),
                       ::testing::ValuesIn(pad_ends_3d),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(0),
                       ::testing::Values(ov::op::RoundingType::CEIL),
                       ::testing::Values(ov::op::PadType::SAME_LOWER));

INSTANTIATE_TEST_SUITE_P(
    smoke_MaxPoolv8_SameLowerPad_CeilRounding_5Dinput,
    MaxPoolingV8LayerTest,
    ::testing::Combine(maxPoolv8_SameLowerPad_CeilRounding_5Dinput_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_5d_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    MaxPoolingV8LayerTest::getTestCaseName);

/* ========= Explicit Pad Ceil Rounding ========== */
const auto maxPoolv8_ExplicitPad_CeilRounding_Params = ::testing::Combine(::testing::ValuesIn(kernels),
                                                                          ::testing::ValuesIn(strides),
                                                                          ::testing::ValuesIn(dilation),
                                                                          ::testing::ValuesIn(pad_begins),
                                                                          ::testing::ValuesIn(pad_ends),
                                                                          ::testing::Values(ov::element::i32),
                                                                          ::testing::Values(0),
                                                                          ::testing::Values(ov::op::RoundingType::CEIL),
                                                                          ::testing::Values(ov::op::PadType::EXPLICIT));

INSTANTIATE_TEST_SUITE_P(
    smoke_MaxPoolv8_ExplicitPad_CeilRounding,
    MaxPoolingV8LayerTest,
    ::testing::Combine(maxPoolv8_ExplicitPad_CeilRounding_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    MaxPoolingV8LayerTest::getTestCaseName);

////* ========== Avg and Max Polling Cases ========== */
/* ========== Valid Pad Rounding Not Applicable ========== */
const auto allPools_ValidPad_Params = ::testing::Combine(
    ::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),
    ::testing::ValuesIn(kernels),
    ::testing::ValuesIn(strides),
    ::testing::Values(std::vector<size_t>({0, 0})),
    ::testing::Values(std::vector<size_t>({0, 0})),
    ::testing::Values(
        ov::op::RoundingType::FLOOR),  // placeholder value - Rounding Type not applicable for Valid pad type
    ::testing::Values(ov::op::PadType::VALID),
    ::testing::Values(false)  // placeholder value - exclude pad not applicable for max pooling
);

INSTANTIATE_TEST_SUITE_P(
    smoke_MAX_and_AVGPool_ValidPad,
    PoolingLayerTest,
    ::testing::Combine(allPools_ValidPad_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    PoolingLayerTest::getTestCaseName);

const auto maxPoolv8_ValidPad_Params = ::testing::Combine(
    ::testing::ValuesIn(kernels),
    ::testing::ValuesIn(strides),
    ::testing::ValuesIn(dilation),
    ::testing::Values(std::vector<size_t>({0, 0})),
    ::testing::Values(std::vector<size_t>({0, 0})),
    ::testing::Values(ov::element::i32),
    ::testing::Values(0),
    ::testing::Values(
        ov::op::RoundingType::FLOOR),  // placeholder value - Rounding Type not applicable for Valid pad type
    ::testing::Values(ov::op::PadType::VALID));

INSTANTIATE_TEST_SUITE_P(
    smoke_MAXPoolv8_ValidPad,
    MaxPoolingV8LayerTest,
    ::testing::Combine(maxPoolv8_ValidPad_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    MaxPoolingV8LayerTest::getTestCaseName);

////* ========== Avg Pooling V16 ========== */
/* +========== Explicit Pad Ceil Rounding ========== */
const auto avgPooV16lExplicitPadCeilRoundingParams =
    ::testing::Combine(::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::ValuesIn(dilation),
                       ::testing::ValuesIn(std::vector<std::vector<size_t>>({{0, 0}, {1, 1}, {0, 1}})),
                       ::testing::ValuesIn(std::vector<std::vector<size_t>>({{0, 0}, {1, 1}, {0, 1}})),
                       ::testing::Values(ov::op::RoundingType::CEIL),
                       ::testing::Values(ov::op::PadType::EXPLICIT),
                       ::testing::Values(true, false));

INSTANTIATE_TEST_SUITE_P(
    smoke_AvgPoolV16_ExplicitPad_CeilRounding,
    AvgPoolingV16LayerTest,
    ::testing::Combine(avgPooV16lExplicitPadCeilRoundingParams,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    AvgPoolingV16LayerTest::getTestCaseName);

const std::vector<avgPoolV16LayerTestParams> psV16Params({avgPoolV16LayerTestParams({2, 2},
                                                                                    {2, 2},
                                                                                    {2, 2},
                                                                                    {0, 0},
                                                                                    {0, 0},
                                                                                    ov::op::RoundingType::CEIL,
                                                                                    ov::op::PadType::EXPLICIT,
                                                                                    false),
                                                          avgPoolV16LayerTestParams({7, 7},
                                                                                    {1, 1},
                                                                                    {2, 2},
                                                                                    {0, 0},
                                                                                    {1, 1},
                                                                                    ov::op::RoundingType::CEIL,
                                                                                    ov::op::PadType::EXPLICIT,
                                                                                    false)});

INSTANTIATE_TEST_SUITE_P(smoke_AvgPoolV16_ExplicitPad_CeilRounding_corner,
                         AvgPoolingV16LayerTest,
                         ::testing::Combine(::testing::ValuesIn(psV16Params),
                                            ::testing::ValuesIn(model_types),
                                            ::testing::Values(ov::test::static_shapes_to_test_representation(
                                                input_shapes_explicit_pad_ceil_rounding_corner_static)),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         AvgPoolingV16LayerTest::getTestCaseName);

/* +========== Explicit Pad Floor Rounding ========== */
const auto avgPoolV16ExplicitPadFloorRoundingParams =
    ::testing::Combine(::testing::ValuesIn(kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::ValuesIn(dilation),
                       ::testing::ValuesIn(std::vector<std::vector<size_t>>({{0, 0}, {1, 1}})),
                       ::testing::ValuesIn(std::vector<std::vector<size_t>>({{0, 0}, {1, 1}})),
                       ::testing::Values(ov::op::RoundingType::FLOOR),
                       ::testing::Values(ov::op::PadType::EXPLICIT),
                       ::testing::Values(true, false));

INSTANTIATE_TEST_SUITE_P(
    smoke_AvgPoolV16_ExplicitPad_FloorRounding,
    AvgPoolingV16LayerTest,
    ::testing::Combine(avgPoolV16ExplicitPadFloorRoundingParams,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    AvgPoolingV16LayerTest::getTestCaseName);

/* ========== Explicit Pad Floor Rounding 5D input========== */
const auto avgPoolV16_ExplicitPad_FloorRounding_5Dinput_Params =
    ::testing::Combine(::testing::ValuesIn(kernel_3d),
                       ::testing::ValuesIn(strides_3d),
                       ::testing::Values(dilation_3d[0]),
                       ::testing::ValuesIn(pad_begins_3d),
                       ::testing::ValuesIn(pad_ends_3d),
                       ::testing::Values(ov::op::RoundingType::FLOOR),
                       ::testing::Values(ov::op::PadType::EXPLICIT),
                       ::testing::Values(true, false));

INSTANTIATE_TEST_SUITE_P(
    smoke_AvgPoolV16_ExplicitPad_FloorRounding_5Dinput,
    AvgPoolingV16LayerTest,
    ::testing::Combine(avgPoolV16_ExplicitPad_FloorRounding_5Dinput_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_5d_2_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    AvgPoolingV16LayerTest::getTestCaseName);

/* ========== Same Upper Pad Floor Rounding 5D input========== */
const auto avgPoolV16_SameUpperPad_FloorRounding_5Dinput_Params =
    ::testing::Combine(::testing::ValuesIn(kernel_3d),
                       ::testing::ValuesIn(strides_3d),
                       ::testing::ValuesIn(dilation_3d),
                       ::testing::ValuesIn(pad_begins_3d),
                       ::testing::ValuesIn(pad_ends_3d),
                       ::testing::Values(ov::op::RoundingType::FLOOR),
                       ::testing::Values(ov::op::PadType::SAME_UPPER),
                       ::testing::Values(true));

INSTANTIATE_TEST_SUITE_P(
    smoke_AvgPoolV16_SameUpperPad_FloorRounding_5Dinput,
    AvgPoolingV16LayerTest,
    ::testing::Combine(avgPoolV16_SameUpperPad_FloorRounding_5Dinput_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_5d_2_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    AvgPoolingV16LayerTest::getTestCaseName);

/* ========== Same Lower Pad Ceil Rounding 5D input========== */
const auto avgPoolV16_SameLowerPad_CeilRounding_5Dinput_Params =
    ::testing::Combine(::testing::ValuesIn(kernel_3d),
                       ::testing::ValuesIn(strides_3d),
                       ::testing::ValuesIn(dilation_3d),
                       ::testing::ValuesIn(pad_begins_3d),
                       ::testing::ValuesIn(pad_ends_3d),
                       ::testing::Values(ov::op::RoundingType::CEIL),
                       ::testing::Values(ov::op::PadType::SAME_LOWER),
                       ::testing::Values(true));

INSTANTIATE_TEST_SUITE_P(
    smoke_AvgPoolV16_SameLowerPad_CeilRounding_5Dinput,
    AvgPoolingV16LayerTest,
    ::testing::Combine(avgPoolV16_SameLowerPad_CeilRounding_5Dinput_Params,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_5d_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    AvgPoolingV16LayerTest::getTestCaseName);

}  // namespace
