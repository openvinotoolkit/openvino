// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, detection_output_op)
{
    NodeBuilder::get_ops().register_factory<opset1::DetectionOutput>();
    const auto box_logits = make_shared<op::Parameter>(element::f32, Shape{1, 2 * 1 * 4});
    const auto class_preds = make_shared<op::Parameter>(element::f32, Shape{1, 2 * 32});
    const auto proposals = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2 * 4});
    const auto aux_class_preds = make_shared<op::Parameter>(element::f32, Shape{1, 2 * 2});
    const auto aux_box_pred = make_shared<op::Parameter>(element::f32, Shape{1, 2 * 1 * 4});

    op::DetectionOutputAttrs attrs;
    attrs.num_classes = 32;
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

    auto detection_output = make_shared<opset1::DetectionOutput>(
        box_logits, class_preds, proposals, aux_class_preds, aux_box_pred, attrs);
    NodeBuilder builder(detection_output);
    auto g_detection_output = as_type_ptr<opset1::DetectionOutput>(builder.create());

    const auto do_attrs = detection_output->get_attrs();
    const auto g_do_attrs = g_detection_output->get_attrs();

    EXPECT_EQ(g_do_attrs.num_classes, do_attrs.num_classes);
    EXPECT_EQ(g_do_attrs.background_label_id, do_attrs.background_label_id);
    EXPECT_EQ(g_do_attrs.top_k, do_attrs.top_k);
    EXPECT_EQ(g_do_attrs.variance_encoded_in_target, do_attrs.variance_encoded_in_target);
    EXPECT_EQ(g_do_attrs.keep_top_k, do_attrs.keep_top_k);
    EXPECT_EQ(g_do_attrs.code_type, do_attrs.code_type);
    EXPECT_EQ(g_do_attrs.share_location, do_attrs.share_location);
    EXPECT_EQ(g_do_attrs.nms_threshold, do_attrs.nms_threshold);
    EXPECT_EQ(g_do_attrs.confidence_threshold, do_attrs.confidence_threshold);
    EXPECT_EQ(g_do_attrs.clip_after_nms, do_attrs.clip_after_nms);
    EXPECT_EQ(g_do_attrs.clip_before_nms, do_attrs.clip_before_nms);
    EXPECT_EQ(g_do_attrs.decrease_label_id, do_attrs.decrease_label_id);
    EXPECT_EQ(g_do_attrs.normalized, do_attrs.normalized);
    EXPECT_EQ(g_do_attrs.input_height, do_attrs.input_height);
    EXPECT_EQ(g_do_attrs.input_width, do_attrs.input_width);
    EXPECT_EQ(g_do_attrs.objectness_score, do_attrs.objectness_score);
}
