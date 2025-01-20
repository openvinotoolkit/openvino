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

class BinaryConvolutionV1StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v1::BinaryConvolution> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }

    const op_type::BinaryConvolutionMode mode = op_type::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
};

TEST_F(BinaryConvolutionV1StaticShapeInferenceTest, default_ctor) {
    op = make_op();
    op->set_strides({1, 1});
    op->set_dilations({1, 1});
    op->set_pads_begin({2, 2});
    op->set_pads_end({2, 1});
    op->set_auto_pad(op::PadType::VALID);

    input_shapes = StaticShapeVector{{1, 3, 10, 12}, {2, 3, 5, 5}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 2, 6, 8}));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({0, 0}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({0, 0}));
}

TEST_F(BinaryConvolutionV1StaticShapeInferenceTest, default_ctor_three_input_shapes) {
    op = make_op();
    op->set_strides({1, 1});
    op->set_dilations({1, 1});
    op->set_pads_begin({2, 2});
    op->set_pads_end({2, 1});
    op->set_auto_pad(op::PadType::VALID);

    // Third input shape (bias) can be provided, but is not used
    input_shapes = StaticShapeVector{{1, 3, 10, 12}, {2, 3, 5, 5}, {2}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 2, 6, 8}));
    EXPECT_EQ(shape_infer->get_pads_begin(), CoordinateDiff({0, 0}));
    EXPECT_EQ(shape_infer->get_pads_end(), CoordinateDiff({0, 0}));
}

TEST_F(BinaryConvolutionV1StaticShapeInferenceTest, auto_pads_same_lower_inputs_dynamic_rank) {
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1, 1};
    const auto pads_begin = CoordinateDiff{0, 0};
    const auto pads_end = CoordinateDiff{0, 0};
    const auto auto_pad = op::PadType::SAME_LOWER;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto filters = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());

    op = make_op(data, filters, strides, pads_begin, pads_end, dilations, mode, pad_value, auto_pad);

    input_shapes = StaticShapeVector{{3, 6, 5, 5}, {7, 6, 3, 3}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({3, 7, 5, 5}));
}

TEST_F(BinaryConvolutionV1StaticShapeInferenceTest, auto_pad_same_lower_inputs_static_ranks) {
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1, 1};
    const auto pads_begin = CoordinateDiff{0, 0};
    const auto pads_end = CoordinateDiff{0, 0};
    const auto auto_pad = op::PadType::SAME_UPPER;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto filters = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));

    op = make_op(data, filters, strides, pads_begin, pads_end, dilations, mode, pad_value, auto_pad);

    input_shapes = StaticShapeVector{{3, 6, 5, 5}, {7, 6, 3, 3}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({3, 7, 5, 5}));
}

TEST_F(BinaryConvolutionV1StaticShapeInferenceTest, data_and_filters_num_channels_not_same) {
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1, 1};
    const auto pads_begin = CoordinateDiff{0, 0};
    const auto pads_end = CoordinateDiff{0, 0};
    const auto auto_pad = op::PadType::SAME_UPPER;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto filters = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));

    op = make_op(data, filters, strides, pads_begin, pads_end, dilations, mode, pad_value, auto_pad);

    input_shapes = StaticShapeVector{{3, 5, 5, 5}, {7, 6, 3, 3}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Data batch channel count (5) does not match filter"));
}

TEST_F(BinaryConvolutionV1StaticShapeInferenceTest, data_rank_not_4) {
    const auto strides = Strides{1, 1};
    const auto dilations = Strides{1, 1};
    const auto pads_begin = CoordinateDiff{0, 0};
    const auto pads_end = CoordinateDiff{0, 0};
    const auto auto_pad = op::PadType::SAME_LOWER;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto filters = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));

    op = make_op(data, filters, strides, pads_begin, pads_end, dilations, mode, pad_value, auto_pad);

    input_shapes = StaticShapeVector{{3, 6, 5}, {7, 6, 3}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Expected 4D for the input. Got:"));
}
