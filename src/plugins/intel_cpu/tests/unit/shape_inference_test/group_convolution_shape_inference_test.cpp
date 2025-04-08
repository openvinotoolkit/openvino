// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "group_convolution_shape_inference.hpp"
#include "openvino/opsets/opset11.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class GroupConvolutionV1StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v1::GroupConvolution> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(GroupConvolutionV1StaticShapeInferenceTest, default_ctor_direct_infer_call) {
    op = make_op();
    op->set_strides({1, 1});
    op->set_dilations({1, 1});
    op->set_auto_pad(op::PadType::EXPLICIT);

    auto pads_begin = CoordinateDiff{2, 2};
    auto pads_end = CoordinateDiff{2, 1};

    input_shapes = StaticShapeVector{{1, 6, 10, 12}, {3, 2, 2, 5, 5}};
    output_shapes = ov::op::v1::shape_infer(op.get(), input_shapes, pads_begin, pads_end);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 6, 10, 11}));
    EXPECT_EQ(pads_begin, CoordinateDiff({2, 2}));
    EXPECT_EQ(pads_end, CoordinateDiff({2, 1}));
}

TEST_F(GroupConvolutionV1StaticShapeInferenceTest, default_ctor) {
    op = make_op();
    op->set_strides({1, 1});
    op->set_dilations({1, 1});
    op->set_pads_begin({2, 2});
    op->set_pads_end({2, 1});
    op->set_auto_pad(op::PadType::EXPLICIT);

    input_shapes = StaticShapeVector{{1, 6, 10, 12}, {3, 2, 2, 5, 5}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 6, 10, 11}));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({2, 2}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({2, 1}));
}

TEST_F(GroupConvolutionV1StaticShapeInferenceTest, default_ctor_three_input_shapes) {
    op = make_op();
    op->set_strides({1, 1});
    op->set_dilations({1, 1});
    op->set_pads_begin({2, 2});
    op->set_pads_end({2, 1});
    op->set_auto_pad(op::PadType::EXPLICIT);

    // Third input shape (bias) can be provided, but is not used
    input_shapes = StaticShapeVector{{1, 6, 10, 12}, {3, 2, 2, 5, 5}, {3}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 6, 10, 11}));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({2, 2}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({2, 1}));
}

TEST_F(GroupConvolutionV1StaticShapeInferenceTest, 1d_explicit_pads_inputs_static_rank) {
    const auto strides = Strides{1};
    const auto dilations = Strides{1};
    const auto pads_begin = CoordinateDiff{0};
    const auto pads_end = CoordinateDiff{0};
    const auto auto_pad = op::PadType::EXPLICIT;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto filters = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));

    op = make_op(data, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    input_shapes = StaticShapeVector{{1, 12, 20}, {12, 1, 1, 3}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({1, 12, 18}));
}

TEST_F(GroupConvolutionV1StaticShapeInferenceTest, 2d_auto_pads_same_lower_inputs_dynamic_rank) {
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1, 1};
    const auto pads_begin = CoordinateDiff{0, 0};
    const auto pads_end = CoordinateDiff{0, 0};
    const auto auto_pad = op::PadType::SAME_LOWER;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto filters = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());

    op = make_op(data, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    input_shapes = StaticShapeVector{{1, 4, 5, 5}, {2, 1, 2, 3, 3}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({1, 2, 5, 5}));
}

TEST_F(GroupConvolutionV1StaticShapeInferenceTest, 3d_auto_pad_same_lower_inputs_static_ranks) {
    const auto strides = Strides{1, 1, 1};
    const auto dilations = Strides{1, 1, 1};
    const auto pads_begin = CoordinateDiff{0, 0, 0};
    const auto pads_end = CoordinateDiff{0, 0, 0};
    const auto auto_pad = op::PadType::SAME_UPPER;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(5));
    const auto filters = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(6));

    op = make_op(data, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    input_shapes = StaticShapeVector{{3, 6, 5, 5, 5}, {1, 6, 6, 3, 3, 3}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({3, 6, 5, 5, 5}));
}

TEST_F(GroupConvolutionV1StaticShapeInferenceTest, dilations_not_defined_for_spatial_shape) {
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1};
    const auto pads_begin = CoordinateDiff{0, 0};
    const auto pads_end = CoordinateDiff{0, 0};
    const auto auto_pad = op::PadType::SAME_LOWER;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto filters = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());

    input_shapes = StaticShapeVector{{1, 4, 5, 5}, {2, 1, 2, 3, 3}};
    OV_EXPECT_THROW(op = make_op(data, filters, strides, pads_begin, pads_end, dilations, auto_pad),
                    NodeValidationFailure,
                    HasSubstr("Dilations should be defined for all and only spatial dimensions"));
}
