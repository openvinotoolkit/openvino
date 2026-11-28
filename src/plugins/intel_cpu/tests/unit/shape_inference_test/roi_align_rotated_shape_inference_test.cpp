// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/roi_align_rotated.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class StaticShapeROIAlignRotatedTest : public OpStaticShapeInferenceTest<op::v15::ROIAlignRotated> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(StaticShapeROIAlignRotatedTest, default_ctor_no_args) {
    op = make_op();
    op->set_pooled_h(2);
    op->set_pooled_w(2);

    input_shapes = StaticShapeVector{{2, 3, 5, 5}, {7, 5}, {7}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{7, 3, 2, 2}));
}

TEST_F(StaticShapeROIAlignRotatedTest, all_inputs_dynamic_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto rois = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto batch_indices = std::make_shared<op::v0::Parameter>(element::i8, PartialShape::dynamic());

    op = make_op(data, rois, batch_indices, 2, 2, 2, 1.0f, false);

    input_shapes = StaticShapeVector{{2, 3, 5, 5}, {10, 5}, {10}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{10, 3, 2, 2}));
}

TEST_F(StaticShapeROIAlignRotatedTest, all_inputs_static_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto rois = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(2));
    const auto batch_indices = std::make_shared<op::v0::Parameter>(element::i8, PartialShape::dynamic(1));

    op = make_op(data, rois, batch_indices, 3, 4, 2, 1.0f, true);

    input_shapes = StaticShapeVector{{2, 8, 5, 5}, {10, 5}, {10}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{10, 8, 3, 4}));
}

TEST_F(StaticShapeROIAlignRotatedTest, incompatible_input_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto rois = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(2));
    const auto batch_indices = std::make_shared<op::v0::Parameter>(element::i8, PartialShape::dynamic(1));

    op = make_op(data, rois, batch_indices, 2, 2, 2, 1.0f, false);

    input_shapes = StaticShapeVector{{2, 8, 5}, {10, 5}, {10}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Expected a 4D tensor for the input data"));
}

TEST_F(StaticShapeROIAlignRotatedTest, incompatible_rois_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto rois = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto batch_indices = std::make_shared<op::v0::Parameter>(element::i8, PartialShape::dynamic(1));

    op = make_op(data, rois, batch_indices, 2, 2, 2, 1.0f, false);

    input_shapes = StaticShapeVector{{2, 8, 5, 5}, {10, 5, 1}, {10}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Expected a 2D tensor for the ROIs input"));
}

TEST_F(StaticShapeROIAlignRotatedTest, incompatible_batch_indicies_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto rois = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(2));
    const auto batch_indices = std::make_shared<op::v0::Parameter>(element::i8, PartialShape::dynamic());

    op = make_op(data, rois, batch_indices, 2, 2, 2, 1.0f, false);
    input_shapes = StaticShapeVector{{2, 8, 5, 5}, {10, 5}, {}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Expected a 1D tensor for the batch indices input."));
}

TEST_F(StaticShapeROIAlignRotatedTest, invalid_rois_2nd_dim) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto rois = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(2));
    const auto batch_indices = std::make_shared<op::v0::Parameter>(element::i8, PartialShape::dynamic(1));

    op = make_op(data, rois, batch_indices, 2, 2, 2, 1.0f, false);

    input_shapes = StaticShapeVector{{2, 8, 5, 5}, {10, 4}, {10}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("op dimension is expected to be equal to 5"));
}

// ROIAlignRotated must not fall back to the generic shape inference, which asks for the data of every
// input (FULL_PORT_MASK): its output shape depends only on the input shapes and the attributes, so the
// registered ShapeInferTA reports an empty port mask instead.
TEST_F(StaticShapeROIAlignRotatedTest, does_not_use_fallback_shape_inference) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto rois = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto batch_indices = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));

    op = make_op(data, rois, batch_indices, 2, 2, 2, 1.0f, false);

    EXPECT_EQ(make_shape_inference(op)->get_port_mask(), EMPTY_PORT_MASK);
}
