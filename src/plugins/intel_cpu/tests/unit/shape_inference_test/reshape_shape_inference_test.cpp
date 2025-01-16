// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class ReshapeV1StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v1::Reshape> {};

TEST_F(ReshapeV1StaticShapeInferenceTest, default_ctor_no_args) {
    op = make_op();
    op->set_special_zero(true);

    int64_t shape_pattern[] = {2, 4, 0, 1, -1};
    auto const_data = std::unordered_map<size_t, ov::Tensor>{{1, Tensor(element::i64, ov::Shape{5}, shape_pattern)}};
    input_shapes = StaticShapeVector{{2, 9, 12, 8}, {5}};

    output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 4, 12, 1, 18}));
}

TEST_F(ReshapeV1StaticShapeInferenceTest, all_inputs_are_dynamic_rank) {
    int64_t shape_pattern[] = {2, 4, 0, 1, -1};
    auto const_data = std::unordered_map<size_t, ov::Tensor>{{1, Tensor(element::i64, ov::Shape{5}, shape_pattern)}};

    const auto data = std::make_shared<op::v0::Parameter>(element::i16, PartialShape::dynamic());
    const auto pattern = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    op = make_op(data, pattern, true);

    input_shapes = StaticShapeVector{{9, 24, 8}, {5}};
    output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 4, 8, 1, 27}));
}

TEST_F(ReshapeV1StaticShapeInferenceTest, all_inputs_are_static_rank) {
    int64_t shape_pattern[] = {2, 4, 1, -1};
    auto const_data = std::unordered_map<size_t, ov::Tensor>{{1, Tensor(element::i64, ov::Shape{4}, shape_pattern)}};

    const auto data = std::make_shared<op::v0::Parameter>(element::i16, PartialShape::dynamic(5));
    const auto pattern = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));
    op = make_op(data, pattern, false);

    input_shapes = StaticShapeVector{{9, 24, 8}, {4}};
    output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 4, 1, 216}));
}

TEST_F(ReshapeV1StaticShapeInferenceTest, pattern_with_special_values) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto pattern = op::v0::Constant::create(element::i32, ov::Shape{2}, {0, -1});

    op = make_op(data, pattern, true);

    input_shapes = StaticShapeVector{{3, 6, 5, 5}, {2}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.front(), StaticShape({3, 150}));
}

TEST_F(ReshapeV1StaticShapeInferenceTest, reshape_to_empty_volume) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 2, 2});
    const auto pattern = op::v0::Constant::create(element::i32, ov::Shape{2}, {0, 4});

    op = make_op(data, pattern, false);

    input_shapes = StaticShapeVector{{0, 2, 2}, {2}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.front(), StaticShape({0, 4}));
}

TEST_F(ReshapeV1StaticShapeInferenceTest, reshape_pattern_not_defined) {
    const auto data = std::make_shared<op::v0::Parameter>(element::i16, PartialShape::dynamic());
    const auto pattern = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    op = make_op(data, pattern, true);

    input_shapes = StaticShapeVector{{9, 24, 8}, {5}};
    OV_EXPECT_THROW(std::ignore = shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Static shape inference lacks constant data on port 1"));
}

TEST_F(ReshapeV1StaticShapeInferenceTest, shape_pattern_as_constant) {
    const auto data = std::make_shared<op::v0::Parameter>(element::i16, PartialShape::dynamic(5));
    const auto pattern = op::v0::Constant::create(element::i32, ov::Shape{3}, {2, 4, 1});
    op = make_op(data, pattern, false);

    input_shapes = StaticShapeVector{{9, 24, 8}, {4}};
    OV_EXPECT_THROW(std::ignore = shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("is incompatible with input shape"));
}
