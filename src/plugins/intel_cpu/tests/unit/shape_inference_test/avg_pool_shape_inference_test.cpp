// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

template <class TOp>
class AvgPoolCommonStaticShapeInferenceTest : public OpStaticShapeInferenceTest<TOp> {};

TYPED_TEST_SUITE_P(AvgPoolCommonStaticShapeInferenceTest);

TYPED_TEST_P(AvgPoolCommonStaticShapeInferenceTest, default_ctor) {
    this->op = this->make_op();
    this->op->set_strides({1, 1});
    this->op->set_pads_begin({2, 2});
    this->op->set_pads_end({2, 1});
    this->op->set_kernel({3, 2});
    this->op->set_rounding_type(op::RoundingType::FLOOR);
    this->op->set_auto_pad(op::PadType::VALID);

    this->input_shapes = ShapeVector{{1, 3, 10, 12}};
    auto shape_infer = make_shape_inference(this->op);
    const auto input_shape_refs = make_static_shape_refs(this->input_shapes);
    this->output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(this->output_shapes.size(), 1);
    EXPECT_EQ(this->output_shapes.front(), StaticShape({1, 3, 8, 11}));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({0, 0}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({0, 0}));
}

TYPED_TEST_P(AvgPoolCommonStaticShapeInferenceTest, no_auto_pad_round_floor) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape{-1, -1, -1, -1});

    const Strides strides{1, 1};
    const ov::Shape pads_begin{2, 2};
    const ov::Shape pads_end{2, 1};
    const ov::Shape kernel_shape{3, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto pad_type = op::PadType::EXPLICIT;

    this->op = this->make_op(data, strides, pads_begin, pads_end, kernel_shape, false, rounding_mode, pad_type);

    this->input_shapes = ShapeVector{{1, 3, 10, 12}};
    auto shape_infer = make_shape_inference(this->op);
    const auto input_shape_refs = make_static_shape_refs(this->input_shapes);
    this->output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(this->output_shapes.size(), 1);
    EXPECT_EQ(this->output_shapes.front(), StaticShape({1, 3, 12, 14}));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({2, 2}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({2, 1}));
}

TYPED_TEST_P(AvgPoolCommonStaticShapeInferenceTest, auto_padding_same_lower_round_ceil) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());

    const Strides strides{1, 3, 2};
    const ov::Shape pads_begin{2, 2, 1};
    const ov::Shape pads_end{2, 1, 10};
    const ov::Shape kernel_shape{5, 5, 5};
    const auto rounding_mode = op::RoundingType::CEIL;
    const auto pad_type = op::PadType::SAME_LOWER;

    this->op = this->make_op(data, strides, pads_begin, pads_end, kernel_shape, false, rounding_mode, pad_type);

    this->input_shapes = ShapeVector{{1, 3, 10, 12, 20}};
    auto shape_infer = make_shape_inference(this->op);
    const auto input_shape_refs = make_static_shape_refs(this->input_shapes);
    this->output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(this->output_shapes.size(), 1);
    EXPECT_EQ(this->output_shapes.front(), StaticShape({1, 3, 10, 4, 10}));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({2, 1, 2}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({2, 1, 1}));
}

TYPED_TEST_P(AvgPoolCommonStaticShapeInferenceTest, auto_padding_same_upper_round_floor_exclude_pad) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());

    const Strides strides{1, 3, 2};
    const ov::Shape pads_begin{2, 2, 1};
    const ov::Shape pads_end{2, 1, 10};
    const ov::Shape kernel_shape{5, 5, 5};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto pad_type = op::PadType::SAME_UPPER;

    this->op = this->make_op(data, strides, pads_begin, pads_end, kernel_shape, true, rounding_mode, pad_type);

    this->input_shapes = ShapeVector{{1, 3, 10, 12, 20}};
    auto shape_infer = make_shape_inference(this->op);
    const auto input_shape_refs = make_static_shape_refs(this->input_shapes);
    this->output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(this->output_shapes.size(), 1);
    EXPECT_EQ(this->output_shapes.front(), StaticShape({1, 3, 10, 4, 10}));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({2, 1, 1}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({2, 1, 2}));
}

TYPED_TEST_P(AvgPoolCommonStaticShapeInferenceTest, auto_padding_same_upper_round_floor) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());

    const Strides strides{1, 1, 1};
    const ov::Shape pads_begin{0, 0, 0};
    const ov::Shape pads_end{0, 0, 0};
    const ov::Shape kernel_shape{2, 2, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto pad_type = op::PadType::SAME_UPPER;

    this->op = this->make_op(data, strides, pads_begin, pads_end, kernel_shape, true, rounding_mode, pad_type);

    this->input_shapes = ShapeVector{{32, 32, 2, 2, 4}};
    auto shape_infer = make_shape_inference(this->op);
    const auto input_shape_refs = make_static_shape_refs(this->input_shapes);
    this->output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(this->output_shapes.size(), 1);
    EXPECT_EQ(this->output_shapes.front(), StaticShape({32, 32, 2, 2, 4}));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({0, 0, 0}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({1, 1, 1}));
}

REGISTER_TYPED_TEST_SUITE_P(AvgPoolCommonStaticShapeInferenceTest,
                            default_ctor,
                            no_auto_pad_round_floor,
                            auto_padding_same_lower_round_ceil,
                            auto_padding_same_upper_round_floor_exclude_pad,
                            auto_padding_same_upper_round_floor);

using AvgPoolOpTypes = Types<ov::op::v1::AvgPool, ov::op::v14::AvgPool>;
INSTANTIATE_TYPED_TEST_SUITE_P(StaticShapeInferenceTest, AvgPoolCommonStaticShapeInferenceTest, AvgPoolOpTypes);

class AvgPoolV14StaticShapeInferenceTest : public OpStaticShapeInferenceTest<ov::op::v14::AvgPool> {};

TEST_F(AvgPoolV14StaticShapeInferenceTest, explicit_padding_ceil_torch) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());

    const Strides strides{2, 2};
    const ov::Shape pads_begin{1, 1};
    const ov::Shape pads_end{1, 1};
    const ov::Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::CEIL_TORCH;
    const auto pad_type = op::PadType::EXPLICIT;

    this->op = this->make_op(data, strides, pads_begin, pads_end, kernel_shape, true, rounding_mode, pad_type);

    this->input_shapes = ShapeVector{{1, 3, 9, 9}};
    auto shape_infer = make_shape_inference(this->op);
    const auto input_shape_refs = make_static_shape_refs(this->input_shapes);
    this->output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(this->output_shapes.size(), 1);
    EXPECT_EQ(this->output_shapes.front(), StaticShape({1, 3, 5, 5}));
}

TEST_F(AvgPoolV14StaticShapeInferenceTest, explicit_padding_ceil_torch_no_strides) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());

    const Strides strides{1, 1};
    const ov::Shape pads_begin{1, 1};
    const ov::Shape pads_end{1, 1};
    const ov::Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::CEIL_TORCH;
    const auto pad_type = op::PadType::EXPLICIT;

    this->op = this->make_op(data, strides, pads_begin, pads_end, kernel_shape, false, rounding_mode, pad_type);

    this->input_shapes = ShapeVector{{1, 3, 9, 9}};
    auto shape_infer = make_shape_inference(this->op);
    const auto input_shape_refs = make_static_shape_refs(this->input_shapes);
    this->output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(this->output_shapes.size(), 1);
    EXPECT_EQ(this->output_shapes.front(), StaticShape({1, 3, 10, 10}));
}

TEST_F(AvgPoolV14StaticShapeInferenceTest, auto_padding_ceil_torch) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());

    const Strides strides{1, 1};
    const ov::Shape pads_begin{1, 1};
    const ov::Shape pads_end{1, 1};
    const ov::Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::CEIL_TORCH;
    const auto pad_type = op::PadType::SAME_LOWER;

    this->op = this->make_op(data, strides, pads_begin, pads_end, kernel_shape, false, rounding_mode, pad_type);

    this->input_shapes = ShapeVector{{1, 3, 9, 9}};
    auto shape_infer = make_shape_inference(this->op);
    const auto input_shape_refs = make_static_shape_refs(this->input_shapes);
    this->output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(this->output_shapes.size(), 1);
    EXPECT_EQ(this->output_shapes.front(), StaticShape({1, 3, 9, 9}));
}
