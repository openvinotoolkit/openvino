// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "convolution_backprop_shape_inference.hpp"
#include "openvino/opsets/opset11.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class ConvolutionBackpropDataV1StaticShapeInferenceTest
    : public OpStaticShapeInferenceTest<op::v1::ConvolutionBackpropData> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(ConvolutionBackpropDataV1StaticShapeInferenceTest, default_ctor_direct_infer_call) {
    const auto spatial_shape = PartialShape{500, 500};
    op = make_op();
    op->set_strides({2, 2});
    op->set_dilations({1, 1});
    op->set_output_padding({0, 0});
    op->set_auto_pad(op::PadType::EXPLICIT);
    op->set_output_shape(spatial_shape.to_shape());

    auto pads_begin = CoordinateDiff{1, 1};
    auto pads_end = CoordinateDiff{1, 1};

    input_shapes = StaticShapeVector{{1, 20, 224, 224}, {20, 10, 3, 3}, {spatial_shape.size()}};

    output_shapes = ov::op::v1::shape_infer(op.get(), input_shapes, pads_begin, pads_end);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 10, 500, 500}));
    EXPECT_EQ(pads_begin, CoordinateDiff({1, 1}));
    EXPECT_EQ(pads_end, CoordinateDiff({1, 1}));
}

TEST_F(ConvolutionBackpropDataV1StaticShapeInferenceTest, default_ctor_with_output_shape) {
    const auto spatial_shape = PartialShape{500, 500};

    op = make_op();
    op->set_strides({2, 2});
    op->set_dilations({1, 1});
    op->set_pads_begin({1, 1});
    op->set_pads_end({1, 1});
    op->set_output_padding({0, 0});
    op->set_auto_pad(op::PadType::EXPLICIT);
    op->set_output_shape(spatial_shape.to_shape());

    input_shapes = StaticShapeVector{{1, 20, 224, 224}, {20, 10, 3, 3}, {spatial_shape.size()}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 10, 500, 500}));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({1, 1}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({1, 1}));
}

TEST_F(ConvolutionBackpropDataV1StaticShapeInferenceTest, default_ctor) {
    op = make_op();
    op->set_strides({1, 1});
    op->set_dilations({1, 1});
    op->set_pads_begin({2, 2});
    op->set_pads_end({2, 1});
    op->set_output_padding({1, 1});
    op->set_auto_pad(op::PadType::VALID);

    input_shapes = StaticShapeVector{{1, 3, 10, 12}, {3, 3, 5, 5}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 3, 15, 17}));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({0, 0}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({0, 0}));
}

TEST_F(ConvolutionBackpropDataV1StaticShapeInferenceTest, default_ctor_more_inputs) {
    const auto spatial_shape = PartialShape{500, 500};

    op = make_op();
    op->set_strides({2, 2});
    op->set_dilations({1, 1});
    op->set_pads_begin({1, 1});
    op->set_pads_end({1, 1});
    op->set_output_padding({0, 0});
    op->set_auto_pad(op::PadType::EXPLICIT);
    op->set_output_shape(spatial_shape.to_shape());

    input_shapes = StaticShapeVector{{1, 20, 224, 224}, {20, 10, 3, 3}, {spatial_shape.size()}, {0}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 10, 500, 500}));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({1, 1}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({1, 1}));
}

TEST_F(ConvolutionBackpropDataV1StaticShapeInferenceTest, 2d_inputs_dynamic_rank_no_spatial_shape) {
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1, 1};
    const auto pads_begin = CoordinateDiff{0, 0};
    const auto pads_end = CoordinateDiff{0, 0};
    const auto auto_pad = op::PadType::SAME_LOWER;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto filters = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());

    op = make_op(data, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    input_shapes = StaticShapeVector{{3, 6, 5, 5}, {6, 1, 3, 3}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({3, 1, 7, 7}));
}

TEST_F(ConvolutionBackpropDataV1StaticShapeInferenceTest, 3d_auto_pad_same_lower_out_spatial_as_const) {
    const auto strides = Strides{1, 1, 1};
    const auto dilations = Strides{1, 1, 1};
    const auto pads_begin = CoordinateDiff{0, 0, 0};
    const auto pads_end = CoordinateDiff{0, 0, 0};
    const auto auto_pad = op::PadType::SAME_UPPER;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(5));
    const auto filters = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(5));
    const auto out_spatial = op::v0::Constant::create(element::i64, ov::Shape{3}, {2, 1, 3});

    op = make_op(data, filters, out_spatial, strides, pads_begin, pads_end, dilations, auto_pad);

    input_shapes = StaticShapeVector{{3, 6, 5, 5, 5}, {6, 2, 3, 3, 3}, {3}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({3, 2, 2, 1, 3}));
}

TEST_F(ConvolutionBackpropDataV1StaticShapeInferenceTest, 3d_auto_pad_same_upper_out_spatial_in_map) {
    const auto strides = Strides{1, 1, 1};
    const auto dilations = Strides{1, 1, 1};
    const auto pads_begin = CoordinateDiff{0, 0, 0};
    const auto pads_end = CoordinateDiff{0, 0, 0};
    const auto auto_pad = op::PadType::SAME_UPPER;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(5));
    const auto filters = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(5));
    const auto out_spatial = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));

    op = make_op(data, filters, out_spatial, strides, pads_begin, pads_end, dilations, auto_pad);
    int32_t spatial_dims[] = {2, 6, 1};
    const auto const_map = std::unordered_map<size_t, ov::Tensor>{{2, {element::i32, ov::Shape{3}, spatial_dims}}};

    input_shapes = StaticShapeVector{{3, 5, 5, 5, 5}, {5, 7, 3, 3, 3}, {3}};
    output_shapes = shape_inference(op.get(), input_shapes, const_map);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({3, 7, 2, 6, 1}));
}
