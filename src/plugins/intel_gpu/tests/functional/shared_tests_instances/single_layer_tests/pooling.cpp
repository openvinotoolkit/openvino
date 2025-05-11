// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/pooling.hpp"

namespace {
using ov::test::PoolingLayerTest;
using ov::test::MaxPoolingV8LayerTest;

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16
};

const std::vector<ov::element::Type> netPrecisions_fp_i32 = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32
};

const std::vector<std::vector<size_t >> kernels = {{3, 3},
                                                   {3, 5}};
const std::vector<std::vector<size_t >> strides = {{1, 1},
                                                   {1, 2}};
const std::vector<std::vector<size_t >> dilations = {{1, 1},
                                                     {1, 2}};
const std::vector<std::vector<size_t >> padBegins = {{0, 0},
                                                     {0, 2}};
const std::vector<std::vector<size_t >> padEnds = {{0, 0},
                                                   {0, 2}};
const std::vector<ov::op::RoundingType> roundingTypes = {ov::op::RoundingType::CEIL,
                                                         ov::op::RoundingType::FLOOR};
const std::vector<ov::element::Type> indexElementTypes = {ov::element::i32};

const std::vector<int64_t> axes = {0, 2};
const std::vector<ov::Shape> inputShapeSmall = {{1, 3, 30, 30}};
const std::vector<ov::Shape> inputShapeLarge = {{1, 3, 50, 50}};

////* ========== Max Pooling ========== */
/* +========== Explicit Pad Floor Rounding ========== */
const auto maxPool_ExplicitPad_FloorRounding_Params = ::testing::Combine(
        ::testing::Values(ov::test::utils::PoolingTypes::MAX),
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::Values(ov::op::RoundingType::FLOOR),
        ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::Values(false)  // placeholder value - exclude pad not applicable for max pooling
);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool_ExplicitPad_FloorRounding, PoolingLayerTest,
                        ::testing::Combine(
                                maxPool_ExplicitPad_FloorRounding_Params,
                                ::testing::ValuesIn(netPrecisions_fp_i32),
                                ::testing::Values(ov::test::static_shapes_to_test_representation(inputShapeLarge)),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        PoolingLayerTest::getTestCaseName);

/* ========== Explicit Pad Ceil Rounding ========== */
const auto maxPool_ExplicitPad_CeilRounding_Params = ::testing::Combine(
        ::testing::Values(ov::test::utils::PoolingTypes::MAX),
        ::testing::ValuesIn(kernels),
        // TODO: Non 1 strides fails in ngraph reference implementation with error "The end corner is out of bounds at axis 3" thrown in the test body.
        ::testing::Values(std::vector<size_t>({1, 1})),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::Values(ov::op::RoundingType::CEIL),
        ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::Values(false)  // placeholder value - exclude pad not applicable for max pooling
);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool_ExplicitPad_CeilRounding, PoolingLayerTest,
                        ::testing::Combine(
                                maxPool_ExplicitPad_CeilRounding_Params,
                                ::testing::ValuesIn(netPrecisions_fp_i32),
                                ::testing::Values(ov::test::static_shapes_to_test_representation(inputShapeLarge)),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        PoolingLayerTest::getTestCaseName);


////* ========== Avg Pooling ========== */
/* +========== Explicit Pad Ceil Rounding ========== */
const auto avgPoolExplicitPadCeilRoundingParams = ::testing::Combine(
        ::testing::Values(ov::test::utils::PoolingTypes::AVG),
        ::testing::ValuesIn(kernels),
        // TODO: Non 1 strides fails in ngraph reference implementation with error "The end corner is out of bounds at axis 3" thrown in the test body.
        ::testing::Values(std::vector<size_t>({1, 1})),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::Values(ov::op::RoundingType::CEIL),
        ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::Values(true, false)
);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_ExplicitPad_CeilRounding, PoolingLayerTest,
                       ::testing::Combine(
                               avgPoolExplicitPadCeilRoundingParams,
                               ::testing::ValuesIn(netPrecisions),
                               ::testing::Values(ov::test::static_shapes_to_test_representation(inputShapeSmall)),
                               ::testing::Values(ov::test::utils::DEVICE_GPU)),
                       PoolingLayerTest::getTestCaseName);

/* +========== Explicit Pad Floor Rounding ========== */
const auto avgPoolExplicitPadFloorRoundingParams = ::testing::Combine(
        ::testing::Values(ov::test::utils::PoolingTypes::AVG),
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::Values(ov::op::RoundingType::FLOOR),
        ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::Values(true, false)
);


INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_ExplicitPad_FloorRounding, PoolingLayerTest,
                        ::testing::Combine(
                                avgPoolExplicitPadFloorRoundingParams,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::static_shapes_to_test_representation(inputShapeSmall)),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        PoolingLayerTest::getTestCaseName);

////* ========== Avg and Max Pooling Cases ========== */
/*    ========== Valid Pad Rounding Not Applicable ========== */
const auto allPools_ValidPad_Params = ::testing::Combine(
        ::testing::Values(ov::test::utils::PoolingTypes::MAX, ov::test::utils::PoolingTypes::AVG),
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::Values(std::vector<size_t>({0, 0})),
        ::testing::ValuesIn(padEnds),
        ::testing::Values(ov::op::RoundingType::FLOOR),  // placeholder value - Rounding Type not applicable for Valid pad type
        // TODO: PadType::VALID seems not to ignore padBegins
        ::testing::Values(ov::op::PadType::VALID),
        ::testing::Values(false)  // placeholder value - exclude pad not applicable for max pooling
);

INSTANTIATE_TEST_SUITE_P(smoke_MAX_and_AVGPool_ValidPad, PoolingLayerTest,
                        ::testing::Combine(
                                allPools_ValidPad_Params,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::static_shapes_to_test_representation(inputShapeLarge)),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        PoolingLayerTest::getTestCaseName);



////* ========== MaxPool v8 ========== */
///* +========== Explicit Pad Floor Rounding ========== */
const auto maxPool8_ExplicitPad_FloorRounding_Params = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::ValuesIn(indexElementTypes),
        ::testing::ValuesIn(axes),
        ::testing::Values(ov::op::RoundingType::FLOOR),
        ::testing::Values(ov::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool8_ExplicitPad_FloorRounding, MaxPoolingV8LayerTest,
                        ::testing::Combine(
                                maxPool8_ExplicitPad_FloorRounding_Params,
                                ::testing::ValuesIn(netPrecisions_fp_i32),
                                ::testing::Values(ov::test::static_shapes_to_test_representation(inputShapeSmall)),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         MaxPoolingV8LayerTest::getTestCaseName);

/* ========== Explicit Pad Ceil Rounding ========== */
const auto maxPool8_ExplicitPad_CeilRounding_Params = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::Values(std::vector<size_t>({1, 1})),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::ValuesIn(indexElementTypes),
        ::testing::ValuesIn(axes),
        ::testing::Values(ov::op::RoundingType::CEIL),
        ::testing::Values(ov::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool8_ExplicitPad_CeilRounding, MaxPoolingV8LayerTest,
                         ::testing::Combine(
                                 maxPool8_ExplicitPad_CeilRounding_Params,
                                 ::testing::ValuesIn(netPrecisions_fp_i32),
                                 ::testing::Values(ov::test::static_shapes_to_test_representation(inputShapeSmall)),
                                 ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         MaxPoolingV8LayerTest::getTestCaseName);

}  // namespace
