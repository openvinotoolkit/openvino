// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/parameter.hpp"
#include "shape_inference/shape_inference.hpp"
#include "shape_inference/static_shape.hpp"

using namespace ov;
using namespace ov::intel_cpu;

template <class T>
class StaticShapeInferenceTest_BEA : public testing::Test {};

// StaticShapeInferenceTest for BinaryElementwiseArithmetis (BEA) operations
TYPED_TEST_SUITE_P(StaticShapeInferenceTest_BEA);

TYPED_TEST_P(StaticShapeInferenceTest_BEA, shape_inference_autob_numpy_equal_rank) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto node = std::make_shared<TypeParam>(A, B);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 1, 1, 5}, StaticShape{3, 1, 6, 1}};
    const auto static_output_shapes = shape_inference(node.get(), static_input_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 1, 6, 5}));
}

TYPED_TEST_P(StaticShapeInferenceTest_BEA, shape_inference_autob_numpy_a_rank_higher) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});

    auto node = std::make_shared<TypeParam>(A, B);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 4, 1, 5}, StaticShape{4, 6, 1}};
    const auto static_output_shapes = shape_inference(node.get(), static_input_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 4, 6, 5}));
}

TYPED_TEST_P(StaticShapeInferenceTest_BEA, shape_inference_autob_numpy_b_rank_higher) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto node = std::make_shared<TypeParam>(A, B);

    std::vector<StaticShape> static_input_shapes = {StaticShape{4, 6, 1}, StaticShape{3, 4, 1, 5}};
    const auto static_output_shapes = shape_inference(node.get(), static_input_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 4, 6, 5}));
}

TYPED_TEST_P(StaticShapeInferenceTest_BEA, shape_inference_autob_numpy_incompatible_shapes) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto node = std::make_shared<TypeParam>(A, B);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 4, 6, 5}, StaticShape{2, 4, 6, 5}};

    ASSERT_THROW(shape_inference(node.get(), static_input_shapes), NodeValidationFailure);
}

TYPED_TEST_P(StaticShapeInferenceTest_BEA, shape_inference_aubtob_none) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto node = std::make_shared<TypeParam>(A, B, op::AutoBroadcastType::NONE);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 4, 6, 5}, StaticShape{3, 4, 6, 5}};
    const auto static_output_shapes = shape_inference(node.get(), static_input_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 4, 6, 5}));
}

TYPED_TEST_P(StaticShapeInferenceTest_BEA, shape_inference_aubtob_none_incompatible_shapes) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto node = std::make_shared<TypeParam>(A, B, op::AutoBroadcastType::NONE);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 4, 6, 5}, StaticShape{3, 1, 6, 1}};

    ASSERT_THROW(shape_inference(node.get(), static_input_shapes), NodeValidationFailure);
}

REGISTER_TYPED_TEST_SUITE_P(StaticShapeInferenceTest_BEA,
                            shape_inference_autob_numpy_equal_rank,
                            shape_inference_autob_numpy_a_rank_higher,
                            shape_inference_autob_numpy_b_rank_higher,
                            shape_inference_autob_numpy_incompatible_shapes,
                            shape_inference_aubtob_none,
                            shape_inference_aubtob_none_incompatible_shapes);

INSTANTIATE_TYPED_TEST_SUITE_P(shape_infer_add, StaticShapeInferenceTest_BEA, ::testing::Types<op::v1::Add>);
INSTANTIATE_TYPED_TEST_SUITE_P(shape_infer_divide, StaticShapeInferenceTest_BEA, ::testing::Types<op::v1::Divide>);
INSTANTIATE_TYPED_TEST_SUITE_P(shape_infer_floor_mod, StaticShapeInferenceTest_BEA, ::testing::Types<op::v1::FloorMod>);
INSTANTIATE_TYPED_TEST_SUITE_P(shape_infer_maximum, StaticShapeInferenceTest_BEA, ::testing::Types<op::v1::Maximum>);
INSTANTIATE_TYPED_TEST_SUITE_P(shape_infer_minimum, StaticShapeInferenceTest_BEA, ::testing::Types<op::v1::Minimum>);
INSTANTIATE_TYPED_TEST_SUITE_P(shape_infer_mod, StaticShapeInferenceTest_BEA, ::testing::Types<op::v1::Mod>);
INSTANTIATE_TYPED_TEST_SUITE_P(shape_infer_multiply, StaticShapeInferenceTest_BEA, ::testing::Types<op::v1::Multiply>);
INSTANTIATE_TYPED_TEST_SUITE_P(shape_infer_power, StaticShapeInferenceTest_BEA, ::testing::Types<op::v1::Power>);
INSTANTIATE_TYPED_TEST_SUITE_P(shape_infer_squared_difference, StaticShapeInferenceTest_BEA, ::testing::Types<op::v0::SquaredDifference>);
INSTANTIATE_TYPED_TEST_SUITE_P(shape_infer_subtract, StaticShapeInferenceTest_BEA, ::testing::Types<op::v1::Subtract>);
