// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;

using Attrs = op::v6::ExperimentalDetectronDetectionOutput::Attributes;
using ExperimentalDetection = op::v6::ExperimentalDetectronDetectionOutput;

TEST(type_prop, detectron_detection_output)
{
    Attrs attrs;
    attrs.class_agnostic_box_regression = false;
    attrs.deltas_weights = {10.0f, 10.0f, 5.0f, 5.0f};
    attrs.max_delta_log_wh = 4.135166645050049f;
    attrs.max_detections_per_image = 100;
    attrs.nms_threshold = 0.5f;
    attrs.num_classes = 81;
    attrs.post_nms_count = 2000;
    attrs.score_threshold = 0.05000000074505806f;

    size_t rois_num = static_cast<size_t>(attrs.max_detections_per_image);

    auto rois = std::make_shared<op::Parameter>(element::f32, Shape{1000, 4});
    auto deltas = std::make_shared<op::Parameter>(element::f32, Shape{1000, 324});
    auto scores = std::make_shared<op::Parameter>(element::f32, Shape{1000, 81});
    auto im_info = std::make_shared<op::Parameter>(element::f32, Shape{1, 3});

    auto detection = std::make_shared<ExperimentalDetection>(rois, deltas, scores, im_info, attrs);

    ASSERT_EQ(detection->get_output_element_type(0), element::f32);
    ASSERT_EQ(detection->get_output_element_type(1), element::i32);
    ASSERT_EQ(detection->get_output_element_type(2), element::f32);

    EXPECT_EQ(detection->get_output_shape(0), (Shape{rois_num, 4}));
    EXPECT_EQ(detection->get_output_shape(1), (Shape{rois_num}));
    EXPECT_EQ(detection->get_output_shape(2), (Shape{rois_num}));


    rois = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic(2));
    deltas = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic(2));
    scores = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic(2));
    im_info = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic(2));

    detection = std::make_shared<ExperimentalDetection>(rois, deltas, scores, im_info, attrs);

    ASSERT_EQ(detection->get_output_element_type(0), element::f32);
    ASSERT_EQ(detection->get_output_element_type(1), element::i32);
    ASSERT_EQ(detection->get_output_element_type(2), element::f32);

    EXPECT_EQ(detection->get_output_shape(0), (Shape{rois_num, 4}));
    EXPECT_EQ(detection->get_output_shape(1), (Shape{rois_num}));
    EXPECT_EQ(detection->get_output_shape(2), (Shape{rois_num}));



}

TEST(type_prop, detectron_detection_output_dynamic_input_shapes)
{
    Attrs attrs;
    attrs.class_agnostic_box_regression = false;
    attrs.deltas_weights = {10.0f, 10.0f, 5.0f, 5.0f};
    attrs.max_delta_log_wh = 4.135166645050049f;
    attrs.max_detections_per_image = 100;
    attrs.nms_threshold = 0.5f;
    attrs.num_classes = 81;
    attrs.post_nms_count = 2000;
    attrs.score_threshold = 0.05000000074505806f;

    size_t rois_num = static_cast<size_t>(attrs.max_detections_per_image);

    struct ShapesAndAttrs
    {
        PartialShape rois_shape;
        PartialShape deltas_shape;
        PartialShape scores_shape;
        PartialShape im_info_shape;
    };

    const auto dyn_dim = Dimension::dynamic();
    const auto dyn_shape = PartialShape::dynamic();

    std::vector<ShapesAndAttrs> shapes = {
        {{1000, 4}, {1000, 324}, {1000, 81}, {1, 3}},
        {{1000, 4}, {1000, 324}, {1000, 81}, {1, dyn_dim}},
        {{1000, 4}, {1000, 324}, {1000, 81}, {dyn_dim, 3}},
        {{1000, 4}, {1000, 324}, {1000, 81}, {dyn_dim, dyn_dim}},
        {{dyn_dim, 4}, {dyn_dim, 324}, {dyn_dim, 81}, {1, 3}},
        {{dyn_dim, 4}, {dyn_dim, 324}, {dyn_dim, 81}, {1, dyn_dim}},
        {{dyn_dim, 4}, {dyn_dim, 324}, {dyn_dim, 81}, {dyn_dim, 3}},
        {{dyn_dim, 4}, {dyn_dim, 324}, {dyn_dim, 81}, {dyn_dim, dyn_dim}},
        {{1000, 4}, {1000, 324}, {1000, 81}, dyn_shape},
        {{1000, 4}, {1000, 324}, dyn_shape, {1, 3}},
        {{1000, 4}, {1000, 324}, dyn_shape, dyn_shape},
        {{1000, 4}, dyn_shape, {1000, 81}, {1, 3}},
        {{1000, 4}, dyn_shape, {1000, 81}, dyn_shape},
        {{1000, 4}, dyn_shape, dyn_shape, {1, 3}},
        {{1000, 4}, dyn_shape, dyn_shape, dyn_shape},
        {dyn_shape, {1000, 324}, {1000, 81}, {1, 3}},
        {dyn_shape, {1000, 324}, {1000, 81}, dyn_shape},
        {dyn_shape, {1000, 324}, dyn_shape, {1, 3}},
        {dyn_shape, {1000, 324}, dyn_shape, dyn_shape},
        {dyn_shape, dyn_shape, {1000, 81}, {1, 3}},
        {dyn_shape, dyn_shape, {1000, 81}, dyn_shape},
        {dyn_shape, dyn_shape, dyn_shape, {1, 3}},
        {dyn_shape, dyn_shape, dyn_shape, dyn_shape},
    };

    for (const auto& s : shapes)
    {
        auto rois = std::make_shared<op::Parameter>(element::f32, s.rois_shape);
        auto deltas = std::make_shared<op::Parameter>(element::f32, s.deltas_shape);
        auto scores = std::make_shared<op::Parameter>(element::f32, s.scores_shape);
        auto im_info = std::make_shared<op::Parameter>(element::f32, s.im_info_shape);

        auto detection =
            std::make_shared<ExperimentalDetection>(rois, deltas, scores, im_info, attrs);

        ASSERT_EQ(detection->get_output_element_type(0), element::f32);
        ASSERT_EQ(detection->get_output_element_type(1), element::i32);
        ASSERT_EQ(detection->get_output_element_type(2), element::f32);

        EXPECT_EQ(detection->get_output_shape(0), (Shape{rois_num, 4}));
        EXPECT_EQ(detection->get_output_shape(1), (Shape{rois_num}));
        EXPECT_EQ(detection->get_output_shape(2), (Shape{rois_num}));
    }
}
