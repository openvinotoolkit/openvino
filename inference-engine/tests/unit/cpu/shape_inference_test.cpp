// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/parameter.hpp>
#include <convolution_shape_inference.hpp>
#include <detection_output_shape_inference.hpp>
#include <openvino/op/ops.hpp>
#include "utils/shape_inference/static_shape.hpp"

using namespace ov;

TEST(StaticShapeInferenceTest, ConvolutionTest) {
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto filters = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto conv =
            std::make_shared<op::v1::Convolution>(data, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    std::vector<PartialShape> input_shapes = {PartialShape{3, 6, 5, 5}, PartialShape{7, 6, 3, 3}}, output_shapes = {PartialShape{}};
    shape_infer(conv.get(), input_shapes, output_shapes);

    ASSERT_EQ(output_shapes[0], PartialShape({3, 7, 5, 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 6, 5, 5}, StaticShape{7, 6, 3, 3}}, static_output_shapes = {StaticShape{}};
    shape_infer(conv.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 7, 5, 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

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

#if 0
TEST(StaticShapeInferenceTest, ConvolutionTimeTest) {
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{3, 6, 5, 5});
    auto filters = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{7, 6, 3, 3});
    auto conv =
            std::make_shared<op::v1::Convolution>(data, filters, strides, pads_begin, pads_end, dilations, auto_pad);
    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 6, 5, 5}, StaticShape{7, 6, 3, 3}}, static_output_shapes = {StaticShape{}};

    auto before = std::chrono::high_resolution_clock::now();
    auto after = std::chrono::high_resolution_clock::now();

    std::cout << conv << std::endl;
    auto convolution_time_sum = 0;
    for (size_t i = 0; i < 10; ++i) {
        before = std::chrono::high_resolution_clock::now();
        shape_infer(conv.get(), static_input_shapes, static_output_shapes);
        after = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
        std::cout << diff << " ns" << std::endl;
        convolution_time_sum += diff;
    }

    // other operation creation and time measurements: ReLU is an example
    auto relu = std::make_shared<op::v0::Relu>(data);
    std::cout << relu << std::endl;
    auto other_op_time_sum = 0;
    for (size_t i = 0; i < 10; ++i) {
        before = std::chrono::high_resolution_clock::now();
        relu->validate_and_infer_types();
        after = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
        std::cout << diff << " ns" << std::endl;
        other_op_time_sum += diff;
    }
    std::cout << (convolution_time_sum >= other_op_time_sum ? "ON PAR WITH CONVOLUTION: " : "LONGER THAN CONVOLUTION ")
              << 1. * other_op_time_sum / convolution_time_sum << std::endl;
}
#endif