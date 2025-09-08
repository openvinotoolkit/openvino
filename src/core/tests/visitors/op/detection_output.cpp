// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/detection_output.hpp"

#include <gtest/gtest.h>

#include "openvino/op/util/attr_types.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

namespace {
void initialize_attributes(ov::op::util::DetectionOutputBase::AttributesBase& attrs) {
    attrs.background_label_id = 0;
    attrs.top_k = 1;
    attrs.variance_encoded_in_target = false;
    attrs.keep_top_k = {1};
    attrs.code_type = string{"caffe.PriorBoxParameter.CORNER"};
    attrs.share_location = true;
    attrs.nms_threshold = 0.64f;
    attrs.confidence_threshold = 1e-4f;
    attrs.clip_after_nms = true;
    attrs.clip_before_nms = false;
    attrs.decrease_label_id = false;
    attrs.normalized = true;
    attrs.input_height = 32;
    attrs.input_width = 32;
    attrs.objectness_score = 0.73f;
}
void is_equal_attrs(const ov::op::util::DetectionOutputBase::AttributesBase& attrs1,
                    const ov::op::util::DetectionOutputBase::AttributesBase& attrs2) {
    EXPECT_EQ(attrs1.background_label_id, attrs2.background_label_id);
    EXPECT_EQ(attrs1.top_k, attrs2.top_k);
    EXPECT_EQ(attrs1.variance_encoded_in_target, attrs2.variance_encoded_in_target);
    EXPECT_EQ(attrs1.keep_top_k, attrs2.keep_top_k);
    EXPECT_EQ(attrs1.code_type, attrs2.code_type);
    EXPECT_EQ(attrs1.share_location, attrs2.share_location);
    EXPECT_EQ(attrs1.nms_threshold, attrs2.nms_threshold);
    EXPECT_EQ(attrs1.confidence_threshold, attrs2.confidence_threshold);
    EXPECT_EQ(attrs1.clip_after_nms, attrs2.clip_after_nms);
    EXPECT_EQ(attrs1.clip_before_nms, attrs2.clip_before_nms);
    EXPECT_EQ(attrs1.decrease_label_id, attrs2.decrease_label_id);
    EXPECT_EQ(attrs1.normalized, attrs2.normalized);
    EXPECT_EQ(attrs1.input_height, attrs2.input_height);
    EXPECT_EQ(attrs1.input_width, attrs2.input_width);
    EXPECT_EQ(attrs1.objectness_score, attrs2.objectness_score);
}
}  // namespace

TEST(attributes, detection_output_op) {
    NodeBuilder::opset().insert<ov::op::v0::DetectionOutput>();
    const auto box_logits = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2 * 1 * 4});
    const auto class_preds = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2 * 32});
    const auto proposals = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2 * 4});
    const auto aux_class_preds = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2 * 2});
    const auto aux_box_pred = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2 * 1 * 4});

    ov::op::v0::DetectionOutput::Attributes attrs;
    initialize_attributes(attrs);
    attrs.num_classes = 32;

    auto detection_output = make_shared<ov::op::v0::DetectionOutput>(box_logits,
                                                                     class_preds,
                                                                     proposals,
                                                                     aux_class_preds,
                                                                     aux_box_pred,
                                                                     attrs);
    NodeBuilder builder(detection_output, {box_logits, class_preds, proposals, aux_class_preds, aux_box_pred});
    auto g_detection_output = ov::as_type_ptr<ov::op::v0::DetectionOutput>(builder.create());

    const auto do_attrs = detection_output->get_attrs();
    const auto g_do_attrs = g_detection_output->get_attrs();

    EXPECT_EQ(g_do_attrs.num_classes, do_attrs.num_classes);
    is_equal_attrs(g_do_attrs, do_attrs);
}

// ------------------------------ V8 ------------------------------
TEST(attributes, detection_output_v8) {
    NodeBuilder::opset().insert<ov::op::v8::DetectionOutput>();
    const auto box_logits = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2 * 1 * 4});
    const auto class_preds = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2 * 32});
    const auto proposals = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2 * 4});
    const auto aux_class_preds = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2 * 2});
    const auto aux_box_pred = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2 * 1 * 4});

    ov::op::v8::DetectionOutput::Attributes attrs;
    initialize_attributes(attrs);

    auto detection_output =
        make_shared<op::v8::DetectionOutput>(box_logits, class_preds, proposals, aux_class_preds, aux_box_pred, attrs);
    NodeBuilder builder(detection_output, {box_logits, class_preds, proposals, aux_class_preds, aux_box_pred});
    auto g_detection_output = ov::as_type_ptr<op::v8::DetectionOutput>(builder.create());

    const auto do_attrs = detection_output->get_attrs();
    const auto g_do_attrs = g_detection_output->get_attrs();

    is_equal_attrs(g_do_attrs, do_attrs);
}
