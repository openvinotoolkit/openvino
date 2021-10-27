// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/shape_inference/static_shape.hpp"
#include <detection_output_shape_inference.hpp>
#include <gtest/gtest.h>
#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>

using namespace ov;

TEST(StaticShapeInferenceTest, DetectionOutputTest) {
    ov::op::v0::DetectionOutput::Attributes attrs;
    attrs.num_classes = 3;
    attrs.background_label_id = -1;
    attrs.top_k = -1;
    attrs.variance_encoded_in_target = true;
    attrs.keep_top_k = {2};
    attrs.code_type = "caffe.PriorBoxParameter.CORNER";
    attrs.share_location = true;
    attrs.nms_threshold = 0.5;
    attrs.confidence_threshold = 0.3;
    attrs.clip_after_nms = false;
    attrs.clip_before_nms = true;
    attrs.decrease_label_id = false;
    attrs.normalized = true;
    attrs.input_height = 0;
    attrs.input_width = 0;
    attrs.objectness_score = 0;

    size_t num_prior_boxes = 2;
    size_t num_loc_classes = attrs.share_location ? 1 : attrs.num_classes;
    size_t prior_box_size = attrs.normalized ? 4 : 5;
    size_t num_images = 2;

    PartialShape loc_shape{-1, -1};
    PartialShape conf_shape{-1, -1};
    PartialShape prior_boxes_shape{-1, -1, -1};

    auto loc = std::make_shared<ov::op::v0::Parameter>(element::f32, loc_shape);
    auto conf = std::make_shared<ov::op::v0::Parameter>(element::f32, conf_shape);
    auto prior_boxes = std::make_shared<ov::op::v0::Parameter>(element::f32, prior_boxes_shape);
    auto det_output = std::make_shared<ov::op::v0::DetectionOutput>(loc, conf, prior_boxes, attrs);

    std::vector<PartialShape> input_shapes = {
        PartialShape{static_cast<int64_t>(num_images),
                     static_cast<int64_t>(num_prior_boxes * num_loc_classes * prior_box_size)},
        PartialShape{static_cast<int64_t>(num_images), static_cast<int64_t>(num_prior_boxes * attrs.num_classes)},
        PartialShape{static_cast<int64_t>(num_images),
                     attrs.variance_encoded_in_target ? 1L : 2L,
                     static_cast<int64_t>(num_prior_boxes * prior_box_size)}};
    std::vector<PartialShape> output_shapes = {ov::PartialShape{}};
    shape_infer(det_output.get(), input_shapes, output_shapes);
    ASSERT_EQ(output_shapes[0], PartialShape({1, 1, static_cast<int64_t>(num_images * attrs.keep_top_k[0]), 7}));

    std::vector<StaticShape> static_input_shapes = {
        StaticShape{num_images, num_prior_boxes * num_loc_classes * prior_box_size},
        StaticShape{num_images, num_prior_boxes * attrs.num_classes},
        StaticShape{num_images, attrs.variance_encoded_in_target ? 1UL : 2UL, num_prior_boxes * prior_box_size}};
    std::vector<StaticShape> static_output_shapes{{0, 0, 0, 0}};
    shape_infer(det_output.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 1, num_images * static_cast<size_t>(attrs.keep_top_k[0]), 7}));
}
