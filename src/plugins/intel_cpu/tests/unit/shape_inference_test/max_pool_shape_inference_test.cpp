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

class MaxPoolV1StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v1::MaxPool> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(MaxPoolV1StaticShapeInferenceTest, default_ctor) {
    op = make_op();
    op->set_strides({1, 1});
    op->set_pads_begin({2, 2});
    op->set_pads_end({2, 1});
    op->set_kernel({3, 2});
    op->set_rounding_type(op::RoundingType::FLOOR);
    op->set_auto_pad(op::PadType::VALID);

    input_shapes = ShapeVector{{1, 3, 10, 12}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 3, 8, 11}));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({0, 0}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({0, 0}));
}

TEST_F(MaxPoolV1StaticShapeInferenceTest, no_auto_pad_round_floor) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape{-1, -1, -1, -1});

    const Strides strides{1, 1};
    const ov::Shape pads_begin{2, 2};
    const ov::Shape pads_end{2, 1};
    const ov::Shape kernel_shape{3, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto pad_type = op::PadType::EXPLICIT;

    op = make_op(data, strides, pads_begin, pads_end, kernel_shape, rounding_mode, pad_type);

    input_shapes = ShapeVector{{1, 3, 10, 12}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 3, 12, 14}));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({2, 2}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({2, 1}));
}

TEST_F(MaxPoolV1StaticShapeInferenceTest, auto_padding_same_lower_round_ceil) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());

    const Strides strides{1, 3, 2};
    const ov::Shape pads_begin{2, 2, 1};
    const ov::Shape pads_end{2, 1, 10};
    const ov::Shape kernel_shape{5, 5, 5};
    const auto rounding_mode = op::RoundingType::CEIL;
    const auto pad_type = op::PadType::SAME_LOWER;

    op = make_op(data, strides, pads_begin, pads_end, kernel_shape, rounding_mode, pad_type);

    input_shapes = ShapeVector{{1, 3, 10, 12, 20}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 3, 10, 4, 10}));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({2, 1, 2}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({2, 1, 1}));
}

class MaxPoolV14StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v14::MaxPool> {
protected:
    void SetUp() override {
        output_shapes.resize(2);
    }
};

TEST_F(MaxPoolV14StaticShapeInferenceTest, ceil_torch_mode_1) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());
    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const Shape pads_begin{1, 1};
    const Shape pads_end{1, 1};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::CEIL_TORCH;

    op = make_op(data, strides, dilations, pads_begin, pads_end, kernel_shape, rounding_mode);
    this->input_shapes = ShapeVector{{1, 3, 5, 5}};
    auto shape_infer = make_shape_inference(this->op);
    const auto input_shape_refs = make_static_shape_refs(this->input_shapes);
    this->output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_THAT(this->output_shapes, Each(StaticShape({1, 3, 3, 3})));
}

TEST_F(MaxPoolV14StaticShapeInferenceTest, ceil_torch_mode_2) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());
    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const Shape pads_begin{1, 1};
    const Shape pads_end{1, 1};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::CEIL_TORCH;

    op = make_op(data, strides, dilations, pads_begin, pads_end, kernel_shape, rounding_mode);
    this->input_shapes = ShapeVector{{1, 3, 9, 9}};
    auto shape_infer = make_shape_inference(this->op);
    const auto input_shape_refs = make_static_shape_refs(this->input_shapes);
    this->output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_THAT(this->output_shapes, Each(StaticShape({1, 3, 5, 5})));
}

template <class TOp>
class MaxPoolCommonStaticShapeInferenceTest : public OpStaticShapeInferenceTest<TOp> {};

TYPED_TEST_SUITE_P(MaxPoolCommonStaticShapeInferenceTest);

TYPED_TEST_P(MaxPoolCommonStaticShapeInferenceTest, default_ctor) {
    this->op = this->make_op();
    this->op->set_strides({1, 1});
    this->op->set_pads_begin({2, 2});
    this->op->set_pads_end({2, 1});
    this->op->set_kernel({3, 2});
    this->op->set_dilations({2, 1});
    this->op->set_rounding_type(op::RoundingType::FLOOR);
    this->op->set_auto_pad(op::PadType::VALID);

    this->input_shapes = ShapeVector{{1, 3, 10, 12}};
    auto shape_infer = make_shape_inference(this->op);
    const auto input_shape_refs = make_static_shape_refs(this->input_shapes);
    this->output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(this->output_shapes.size(), 2);
    EXPECT_THAT(this->output_shapes, Each(StaticShape({1, 3, 6, 11})));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({0, 0}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({0, 0}));
}

TYPED_TEST_P(MaxPoolCommonStaticShapeInferenceTest, no_dilation) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape{-1, -1, -1, -1});

    const Strides strides{1, 1};
    const Strides dilations{1, 1};
    const ov::Shape pads_begin{1, 1};
    const ov::Shape pads_end{0, 0};
    const ov::Shape kernel_shape{2, 2};

    this->op = this->make_op(data, strides, dilations, pads_begin, pads_end, kernel_shape);

    this->input_shapes = ShapeVector{{2, 3, 13, 13}};
    auto shape_infer = make_shape_inference(this->op);
    const auto input_shape_refs = make_static_shape_refs(this->input_shapes);
    this->output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(this->output_shapes.size(), 2);
    EXPECT_THAT(this->output_shapes, Each(StaticShape({2, 3, 13, 13})));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({1, 1}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({0, 0}));
}

TYPED_TEST_P(MaxPoolCommonStaticShapeInferenceTest, with_dilations) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f64, PartialShape::dynamic());

    const Strides strides{1, 1};
    const Strides dilations{2, 3};
    const ov::Shape pads_begin{0, 0};
    const ov::Shape pads_end{1, 1};
    const ov::Shape kernel_shape{2, 2};

    this->op = this->make_op(data, strides, dilations, pads_begin, pads_end, kernel_shape);

    this->input_shapes = ShapeVector{{2, 4, 13, 13}};
    auto shape_infer = make_shape_inference(this->op);
    const auto input_shape_refs = make_static_shape_refs(this->input_shapes);
    this->output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(this->output_shapes.size(), 2);
    EXPECT_THAT(this->output_shapes, Each(StaticShape({2, 4, 12, 11})));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({0, 0}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({1, 1}));
}

REGISTER_TYPED_TEST_SUITE_P(MaxPoolCommonStaticShapeInferenceTest,
                            default_ctor,
                            no_dilation,
                            with_dilations);

using MaxPoolOpTypes = Types<ov::op::v8::MaxPool, ov::op::v14::MaxPool>;
INSTANTIATE_TYPED_TEST_SUITE_P(StaticShapeInferenceTest, MaxPoolCommonStaticShapeInferenceTest, MaxPoolOpTypes);
