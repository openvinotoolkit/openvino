// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "detection_output_shape_inference.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

template <typename T1, typename T2 = typename T1::Attributes>
std::shared_ptr<Node> create_detection_output(const PartialShape& box_logits_shape,
                                              const PartialShape& class_preds_shape,
                                              const PartialShape& proposals_shape,
                                              const PartialShape& aux_class_preds_shape,
                                              const PartialShape& aux_box_preds_shape,
                                              T2& attrs,
                                              element::Type input_type,
                                              element::Type proposals_type) {
    auto box_logits = std::make_shared<op::v0::Parameter>(input_type, box_logits_shape);
    auto class_preds = std::make_shared<op::v0::Parameter>(input_type, class_preds_shape);
    auto proposals = std::make_shared<op::v0::Parameter>(proposals_type, proposals_shape);
    auto aux_class_preds = std::make_shared<op::v0::Parameter>(input_type, aux_class_preds_shape);
    auto aux_box_preds = std::make_shared<op::v0::Parameter>(input_type, aux_box_preds_shape);
    return std::make_shared<T1>(box_logits, class_preds, proposals, aux_class_preds, aux_box_preds, attrs);
}

TEST(StaticShapeInferenceTest, detection_output_v0_top_k) {
    op::v0::DetectionOutput::Attributes attrs;
    attrs.keep_top_k = {-1};
    attrs.top_k = 7;
    attrs.normalized = true;
    attrs.num_classes = 2;
    auto op = create_detection_output<op::v0::DetectionOutput>(PartialShape{4, 20},
                                                               PartialShape{4, 10},
                                                               PartialShape{4, 2, 20},
                                                               PartialShape{4, 10},
                                                               PartialShape{4, 20},
                                                               attrs,
                                                               element::f32,
                                                               element::f32);

    const std::vector<StaticShape> input_shapes = {StaticShape{4, 20},
                                                   StaticShape{4, 10},
                                                   StaticShape{4, 2, 20},
                                                   StaticShape{4, 10},
                                                   StaticShape{4, 20}};
    std::vector<StaticShape> output_shapes = {StaticShape{}};
    output_shapes = shape_inference(op.get(), input_shapes);
    ASSERT_EQ(output_shapes[0], StaticShape({1, 1, 56, 7}));
}

TEST(StaticShapeInferenceTest, detection_output_v0_no_share_location) {
    op::v0::DetectionOutput::Attributes attrs;
    attrs.keep_top_k = {-1};
    attrs.top_k = -1;
    attrs.normalized = true;
    attrs.num_classes = 2;
    attrs.share_location = false;
    auto op = create_detection_output<op::v0::DetectionOutput>(PartialShape{4, 40},
                                                               PartialShape{4, 10},
                                                               PartialShape{4, 2, 20},
                                                               PartialShape{4, 10},
                                                               PartialShape{4, 40},
                                                               attrs,
                                                               element::f32,
                                                               element::f32);

    const std::vector<StaticShape> input_shapes = {StaticShape{4, 40},
                                                   StaticShape{4, 10},
                                                   StaticShape{4, 2, 20},
                                                   StaticShape{4, 10},
                                                   StaticShape{4, 40}};
    std::vector<StaticShape> output_shapes = {StaticShape{}};
    output_shapes = shape_inference(op.get(), input_shapes);
    ASSERT_EQ(output_shapes[0], StaticShape({1, 1, 40, 7}));
}

TEST(StaticShapeInferenceTest, detection_output_v0_basic) {
    op::v0::DetectionOutput::Attributes attrs;
    attrs.keep_top_k = {200};
    attrs.num_classes = 2;
    attrs.normalized = true;
    auto op = create_detection_output<op::v0::DetectionOutput>(PartialShape{4, 20},
                                                               PartialShape{4, 10},
                                                               PartialShape{4, 2, 20},
                                                               PartialShape{4, 10},
                                                               PartialShape{4, 20},
                                                               attrs,
                                                               element::f32,
                                                               element::f32);

    const std::vector<StaticShape> input_shapes = {StaticShape{4, 20},
                                                   StaticShape{4, 10},
                                                   StaticShape{4, 2, 20},
                                                   StaticShape{4, 10},
                                                   StaticShape{4, 20}};
    std::vector<StaticShape> output_shapes = {StaticShape{}};
    output_shapes = shape_inference(op.get(), input_shapes);
    ASSERT_EQ(output_shapes[0], (StaticShape{1, 1, 800, 7}));
}

TEST(StaticShapeInferenceTest, detection_output_v0_default_ctor) {
    auto op = std::make_shared<op::v0::DetectionOutput>();

    op::v0::DetectionOutput::Attributes attrs;
    attrs.keep_top_k = {200};
    attrs.num_classes = 2;
    attrs.normalized = true;
    op->set_attrs(attrs);

    const std::vector<StaticShape> input_shapes = {StaticShape{4, 20},
                                                   StaticShape{4, 10},
                                                   StaticShape{4, 2, 20},
                                                   StaticShape{4, 10},
                                                   StaticShape{4, 20}};
    std::vector<StaticShape> output_shapes = {StaticShape{}};

    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], (StaticShape{1, 1, 800, 7}));
}

TEST(StaticShapeInferenceTest, detection_output_v8_top_k) {
    op::v8::DetectionOutput::Attributes attrs;
    attrs.keep_top_k = {-1};
    attrs.top_k = 7;
    attrs.normalized = true;
    auto op = create_detection_output<op::v8::DetectionOutput>(PartialShape{4, 20},
                                                               PartialShape{4, 10},
                                                               PartialShape{4, 2, 20},
                                                               PartialShape{4, 10},
                                                               PartialShape{4, 20},
                                                               attrs,
                                                               element::f32,
                                                               element::f32);

    const std::vector<StaticShape> input_shapes = {StaticShape{4, 20},
                                                   StaticShape{4, 10},
                                                   StaticShape{4, 2, 20},
                                                   StaticShape{4, 10},
                                                   StaticShape{4, 20}};
    std::vector<StaticShape> output_shapes = {StaticShape{}};
    output_shapes = shape_inference(op.get(), input_shapes);
    ASSERT_EQ(output_shapes[0], StaticShape({1, 1, 56, 7}));
}

TEST(StaticShapeInferenceTest, detection_output_v8_no_share_location) {
    op::v8::DetectionOutput::Attributes attrs;
    attrs.keep_top_k = {-1};
    attrs.top_k = -1;
    attrs.normalized = true;
    attrs.share_location = false;
    auto op = create_detection_output<op::v8::DetectionOutput>(PartialShape{4, 40},
                                                               PartialShape{4, 10},
                                                               PartialShape{4, 2, 20},
                                                               PartialShape{4, 10},
                                                               PartialShape{4, 40},
                                                               attrs,
                                                               element::f32,
                                                               element::f32);

    const std::vector<StaticShape> input_shapes = {StaticShape{4, 40},
                                                   StaticShape{4, 10},
                                                   StaticShape{4, 2, 20},
                                                   StaticShape{4, 10},
                                                   StaticShape{4, 40}};
    std::vector<StaticShape> output_shapes = {StaticShape{}};
    output_shapes = shape_inference(op.get(), input_shapes);
    ASSERT_EQ(output_shapes[0], StaticShape({1, 1, 40, 7}));
}

TEST(StaticShapeInferenceTest, detection_output_v8_basic) {
    op::v8::DetectionOutput::Attributes attrs;
    attrs.keep_top_k = {200};
    attrs.normalized = true;
    auto op = create_detection_output<op::v8::DetectionOutput>(PartialShape{4, 20},
                                                               PartialShape{4, 10},
                                                               PartialShape{4, 2, 20},
                                                               PartialShape{4, 10},
                                                               PartialShape{4, 20},
                                                               attrs,
                                                               element::f32,
                                                               element::f32);

    const std::vector<StaticShape> input_shapes = {StaticShape{4, 20},
                                                   StaticShape{4, 10},
                                                   StaticShape{4, 2, 20},
                                                   StaticShape{4, 10},
                                                   StaticShape{4, 20}};
    std::vector<StaticShape> output_shapes = {StaticShape{}};
    output_shapes = shape_inference(op.get(), input_shapes);
    ASSERT_EQ(output_shapes[0], (StaticShape{1, 1, 800, 7}));
}

TEST(StaticShapeInferenceTest, detection_output_v8_default_ctor) {
    auto op = std::make_shared<op::v8::DetectionOutput>();

    op::v8::DetectionOutput::Attributes attrs;
    attrs.keep_top_k = {200};
    attrs.normalized = true;
    op->set_attrs(attrs);

    const std::vector<StaticShape> input_shapes = {StaticShape{4, 20},
                                                   StaticShape{4, 10},
                                                   StaticShape{4, 2, 20},
                                                   StaticShape{4, 10},
                                                   StaticShape{4, 20}};
    std::vector<StaticShape> output_shapes = {StaticShape{}};

    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], (StaticShape{1, 1, 800, 7}));
}
