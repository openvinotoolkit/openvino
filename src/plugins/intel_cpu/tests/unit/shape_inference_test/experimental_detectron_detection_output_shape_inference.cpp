// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <experimental_detectron_detection_output_shape_inference.hpp>
#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>

#include "utils/shape_inference/static_shape.hpp"

using namespace ov;
using namespace ov::intel_cpu;

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

    auto rois = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto deltas = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto im_info = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});

    auto detection =
        std::make_shared<ov::op::v6::ExperimentalDetectronDetectionOutput>(rois, deltas, scores, im_info, attrs);
    std::vector<PartialShape> input_shapes = {PartialShape::dynamic(),
                                              PartialShape::dynamic(),
                                              PartialShape::dynamic(),
                                              PartialShape::dynamic()};
    std::vector<PartialShape> output_shapes = {PartialShape::dynamic(),
                                               PartialShape::dynamic(),
                                               PartialShape::dynamic()};
    shape_infer(detection.get(), input_shapes, output_shapes);
    ASSERT_EQ(output_shapes[0], (PartialShape{rois_num, 4}));
    ASSERT_EQ(output_shapes[1], (PartialShape{rois_num}));
    ASSERT_EQ(output_shapes[2], (PartialShape{rois_num}));

    input_shapes = {PartialShape{-1, -1}, PartialShape{-1, -1}, PartialShape{-1, -1}, PartialShape{-1, -1}};
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

    std::vector<StaticShape> static_input_shapes = {StaticShape{16, 4},
                                                    StaticShape{16, 8},
                                                    StaticShape{16, 2},
                                                    StaticShape{1, 3}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}, StaticShape{}, StaticShape{}};
    shape_infer(detection.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({attrs.max_detections_per_image, 4}));
    ASSERT_EQ(static_output_shapes[1], StaticShape({attrs.max_detections_per_image}));
    ASSERT_EQ(static_output_shapes[2], StaticShape({attrs.max_detections_per_image}));
}