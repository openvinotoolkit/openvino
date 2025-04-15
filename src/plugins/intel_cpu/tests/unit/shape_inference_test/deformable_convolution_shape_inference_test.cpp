// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/opsets/opset11.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class DeformableConvolutionV8StaticShapeInferenceTest
    : public OpStaticShapeInferenceTest<op::v8::DeformableConvolution> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(DeformableConvolutionV8StaticShapeInferenceTest, default_ctor) {
    op = make_op();
    op->set_strides({1, 2});
    op->set_dilations({1, 2});
    op->set_pads_begin({2, 2});
    op->set_pads_end({2, 1});
    op->set_auto_pad(op::PadType::VALID);
    op->set_group(4);
    op->set_deformable_group(2);

    input_shapes = StaticShapeVector{{1, 4, 5, 5}, {1, 36, 3, 1}, {4, 1, 3, 3}, {1, 18, 3, 1}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 4, 3, 1}));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({0, 0}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({0, 0}));
}

TEST_F(DeformableConvolutionV8StaticShapeInferenceTest, pads_same_lower_inputs_dynamic_rank_no_masks) {
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1, 1};
    const auto pads_begin = CoordinateDiff{0, 0};
    const auto pads_end = CoordinateDiff{0, 0};
    const auto auto_pad = op::PadType::SAME_LOWER;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto offsets = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto filters = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());

    op = make_op(data, offsets, filters, strides, pads_begin, pads_end, dilations, auto_pad, 4, 2);

    input_shapes = StaticShapeVector{{1, 4, 5, 5}, {1, 36, 5, 5}, {4, 1, 3, 3}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({1, 4, 5, 5}));
}

TEST_F(DeformableConvolutionV8StaticShapeInferenceTest, pads_same_lower_inputs_dynamic_rank) {
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1, 1};
    const auto pads_begin = CoordinateDiff{0, 0};
    const auto pads_end = CoordinateDiff{0, 0};
    const auto auto_pad = op::PadType::SAME_LOWER;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto offsets = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto filters = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto masks = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());

    op = make_op(data, offsets, filters, masks, strides, pads_begin, pads_end, dilations, auto_pad, 4, 2);

    input_shapes = StaticShapeVector{{1, 4, 5, 5}, {1, 36, 5, 5}, {4, 1, 3, 3}, {1, 18, 5, 5}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({1, 4, 5, 5}));
}

TEST_F(DeformableConvolutionV8StaticShapeInferenceTest, pads_same_uper_inputs_static_rank_no_masks) {
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1, 1};
    const auto pads_begin = CoordinateDiff{0, 0};
    const auto pads_end = CoordinateDiff{0, 0};
    const auto auto_pad = op::PadType::SAME_UPPER;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto offsets = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto filters = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));

    op = make_op(data, offsets, filters, strides, pads_begin, pads_end, dilations, auto_pad, 4, 2);

    input_shapes = StaticShapeVector{{1, 4, 5, 5}, {1, 36, 5, 5}, {4, 1, 3, 3}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({1, 4, 5, 5}));
}

TEST_F(DeformableConvolutionV8StaticShapeInferenceTest, pads_same_upper_inputs_static_rank) {
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1, 1};
    const auto pads_begin = CoordinateDiff{0, 0};
    const auto pads_end = CoordinateDiff{0, 0};
    const auto auto_pad = op::PadType::SAME_UPPER;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto offsets = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto filters = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto masks = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));

    op = make_op(data, offsets, filters, masks, strides, pads_begin, pads_end, dilations, auto_pad, 4, 2);

    input_shapes = StaticShapeVector{{1, 4, 5, 5}, {1, 36, 5, 5}, {4, 1, 3, 3}, {1, 18, 5, 5}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({1, 4, 5, 5}));
}

TEST_F(DeformableConvolutionV8StaticShapeInferenceTest, mask_channel_dimension_not_divisible_by_deformable_group) {
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1, 1};
    const auto pads_begin = CoordinateDiff{0, 0};
    const auto pads_end = CoordinateDiff{0, 0};
    const auto auto_pad = op::PadType::SAME_UPPER;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto offsets = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto filters = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto masks = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));

    op = make_op(data, offsets, filters, strides, pads_begin, pads_end, dilations, auto_pad, 4, 2);

    input_shapes = StaticShapeVector{{1, 4, 5, 5}, {1, 36, 5, 5}, {4, 1, 3, 3}, {1, 17, 5, 5}};

    OV_EXPECT_THROW(
        shape_inference(op.get(), input_shapes),
        NodeValidationFailure,
        HasSubstr(
            "The channels dimension of mask input is not compatible with filters and 'deformable group' attribute"));
}
