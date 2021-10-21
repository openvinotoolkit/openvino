// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/shape_inference/static_shape.hpp"
#include <convolution_shape_inference.hpp>
#include <experimental_detectron_detection_output_shape_inference.hpp>
#include <gtest/gtest.h>
#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>

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

TEST(StaticShapeInferenceTest, ExperimentalDetectronDetectionOutputTest) {
  using Attrs = op::v6::ExperimentalDetectronDetectionOutput::Attributes;
  Attrs attrs;
  attrs.class_agnostic_box_regression = true;
  attrs.deltas_weights = {10.0f, 10.0f, 5.0f, 5.0f};
  attrs.max_delta_log_wh = 2.0f;
  attrs.max_detections_per_image = 5;
  attrs.nms_threshold = 0.2f;
  attrs.num_classes = 2;
  attrs.post_nms_count = 500;
  attrs.score_threshold = 0.01000000074505806f;
  int64_t rois_num = static_cast<int64_t>(attrs.max_detections_per_image);

  auto rois = std::make_shared<ov::op::v0::Parameter>(element::f32,
                                                      PartialShape{-1, -1});
  auto deltas = std::make_shared<ov::op::v0::Parameter>(element::f32,
                                                        PartialShape{-1, -1});
  auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32,
                                                        PartialShape{-1, -1});
  auto im_info = std::make_shared<ov::op::v0::Parameter>(element::f32,
                                                         PartialShape{-1, -1});

  auto detection =
      std::make_shared<ov::op::v6::ExperimentalDetectronDetectionOutput>(
          rois, deltas, scores, im_info, attrs);
  std::vector<PartialShape> input_shapes = {
      PartialShape::dynamic(), PartialShape::dynamic(), PartialShape::dynamic(),
      PartialShape::dynamic()};
  std::vector<PartialShape> output_shapes = {PartialShape::dynamic(),
                                             PartialShape::dynamic(),
                                             PartialShape::dynamic()};
  shape_infer(detection.get(), input_shapes, output_shapes);
  ASSERT_EQ(output_shapes[0], (PartialShape{rois_num, 4}));
  ASSERT_EQ(output_shapes[1], (PartialShape{rois_num}));
  ASSERT_EQ(output_shapes[2], (PartialShape{rois_num}));

  input_shapes = {PartialShape{-1, -1}, PartialShape{-1, -1},
                  PartialShape{-1, -1}, PartialShape{-1, -1}};
  output_shapes = {PartialShape{}, PartialShape{}, PartialShape{}};
  shape_infer(detection.get(), input_shapes, output_shapes);
  ASSERT_EQ(output_shapes[0], (PartialShape{rois_num, 4}));
  ASSERT_EQ(output_shapes[1], (PartialShape{rois_num}));
  ASSERT_EQ(output_shapes[2], (PartialShape{rois_num}));

  input_shapes = {PartialShape{16, 4}, PartialShape{16, 8}, PartialShape{16, 2}, PartialShape{1, 3}};
  output_shapes = {PartialShape{}, PartialShape{}, PartialShape{}};
  shape_infer(detection.get(), input_shapes, output_shapes);
  ASSERT_EQ(output_shapes[0], (PartialShape{rois_num, 4}));
  ASSERT_EQ(output_shapes[1], (PartialShape{rois_num}));
  ASSERT_EQ(output_shapes[2], (PartialShape{rois_num}));

  std::vector<StaticShape> static_input_shapes = {StaticShape{16, 4}, StaticShape{16, 8}, StaticShape{16, 2}, StaticShape{1, 3}};
  std::vector<StaticShape> static_output_shapes = {StaticShape{}, StaticShape{},
                                                   StaticShape{}};
  shape_infer(detection.get(), static_input_shapes, static_output_shapes);
  ASSERT_EQ(static_output_shapes[0],
            StaticShape({attrs.max_detections_per_image, 4}));
  ASSERT_EQ(static_output_shapes[1], StaticShape({attrs.max_detections_per_image}));
  ASSERT_EQ(static_output_shapes[2], StaticShape({attrs.max_detections_per_image}));
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