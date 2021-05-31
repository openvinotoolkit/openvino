// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;

using ExperimentalProposals = op::v6::ExperimentalDetectronGenerateProposalsSingleImage;
using Attrs = op::v6::ExperimentalDetectronGenerateProposalsSingleImage::Attributes;

TEST(type_prop, detectron_proposals)
{
    Attrs attrs;
    attrs.min_size = 0.0f;
    attrs.nms_threshold = 0.699999988079071f;
    attrs.post_nms_count = 1000;
    attrs.pre_nms_count = 1000;

    size_t post_nms_count = 1000;

    auto im_info = std::make_shared<op::Parameter>(element::f32, Shape{3});
    auto anchors = std::make_shared<op::Parameter>(element::f32, Shape{201600, 4});
    auto deltas = std::make_shared<op::Parameter>(element::f32, Shape{12, 200, 336});
    auto scores = std::make_shared<op::Parameter>(element::f32, Shape{3, 200, 336});

    auto proposals =
        std::make_shared<ExperimentalProposals>(im_info, anchors, deltas, scores, attrs);

    ASSERT_EQ(proposals->get_output_element_type(0), element::f32);
    ASSERT_EQ(proposals->get_output_element_type(1), element::f32);
    EXPECT_EQ(proposals->get_output_shape(0), (Shape{post_nms_count, 4}));
    EXPECT_EQ(proposals->get_output_shape(1), (Shape{post_nms_count}));

    im_info = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic(1));
    anchors = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic(2));
    deltas = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic(3));
    scores = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic(3));

    proposals = std::make_shared<ExperimentalProposals>(im_info, anchors, deltas, scores, attrs);

    ASSERT_EQ(proposals->get_output_element_type(0), element::f32);
    ASSERT_EQ(proposals->get_output_element_type(1), element::f32);
    EXPECT_EQ(proposals->get_output_shape(0), (Shape{post_nms_count, 4}));
    EXPECT_EQ(proposals->get_output_shape(1), (Shape{post_nms_count}));



}

TEST(type_prop, detectron_proposals_dynamic)
{
    struct ShapesAndAttrs
    {
        PartialShape im_info_shape;
        PartialShape anchors_shape;
        PartialShape deltas_shape;
        PartialShape scores_shape;
        size_t post_nms_count;
    };

    const auto dyn_dim = Dimension::dynamic();
    const auto dyn_shape = PartialShape::dynamic();

    std::vector<ShapesAndAttrs> shapes = {
        {{3}, {201600, 4}, {12, 200, 336}, {3, 200, 336}, 1000},
        {{3}, {201600, 4}, {12, 200, 336}, dyn_shape, 500},
        {{3}, {201600, 4}, dyn_shape, {3, 200, 336}, 700},
        {{3}, {201600, 4}, dyn_shape, dyn_shape, 300},
        {{3}, dyn_shape, {12, 200, 336}, {3, 200, 336}, 200},
        {{3}, dyn_shape, {12, 200, 336}, dyn_shape, 40},
        {{3}, dyn_shape, dyn_shape, {3, 200, 336}, 70},
        {{3}, dyn_shape, dyn_shape, dyn_shape, 60},
        {dyn_shape, {201600, 4}, {12, 200, 336}, {3, 200, 336}, 500},
        {dyn_shape, {201600, 4}, {12, 200, 336}, dyn_shape, 400},
        {dyn_shape, {201600, 4}, dyn_shape, {3, 200, 336}, 350},
        {dyn_shape, {201600, 4}, dyn_shape, dyn_shape, 440},
        {dyn_shape, dyn_shape, {12, 200, 336}, {3, 200, 336}, 315},
        {dyn_shape, dyn_shape, {12, 200, 336}, dyn_shape, 130},
        {dyn_shape, dyn_shape, dyn_shape, {3, 200, 336}, 1000},
        {dyn_shape, dyn_shape, dyn_shape, dyn_shape, 700},
        {{3}, {dyn_dim, 4}, {12, 200, 336}, {3, 200, 336}, 540},
        {{3}, {dyn_dim, 4}, {12, 200, 336}, {dyn_dim, 200, 336}, 600},
        {{3}, {dyn_dim, 4}, {dyn_dim, 200, 336}, {3, 200, 336}, 75},
        {{3}, {dyn_dim, 4}, {dyn_dim, 200, 336}, {dyn_dim, 200, 336}, 80},
        {{3}, {201600, 4}, {12, 200, dyn_dim}, {3, 200, dyn_dim}, 430},
        {{3}, {201600, 4}, {12, dyn_dim, 336}, {3, dyn_dim, 336}, 180},
        {{3}, {201600, 4}, {12, dyn_dim, dyn_dim}, {3, dyn_dim, dyn_dim}, 170},
        {{3}, {dyn_dim, 4}, {12, 200, dyn_dim}, {3, 200, dyn_dim}, 200},
        {{3}, {dyn_dim, 4}, {12, dyn_dim, 336}, {3, dyn_dim, 336}, 800},
        {{3}, {dyn_dim, 4}, {12, dyn_dim, dyn_dim}, {3, dyn_dim, dyn_dim}, 560},
    };

    for (const auto& s : shapes)
    {
        Attrs attrs;
        attrs.min_size = 0.0f;
        attrs.nms_threshold = 0.699999988079071f;
        attrs.post_nms_count = static_cast<int64_t>(s.post_nms_count);
        attrs.pre_nms_count = 1000;

        auto im_info = std::make_shared<op::Parameter>(element::f32, s.im_info_shape);
        auto anchors = std::make_shared<op::Parameter>(element::f32, s.anchors_shape);
        auto deltas = std::make_shared<op::Parameter>(element::f32, s.deltas_shape);
        auto scores = std::make_shared<op::Parameter>(element::f32, s.scores_shape);

        auto proposals =
            std::make_shared<ExperimentalProposals>(im_info, anchors, deltas, scores, attrs);

        ASSERT_EQ(proposals->get_output_element_type(0), element::f32);
        ASSERT_EQ(proposals->get_output_element_type(1), element::f32);
        EXPECT_EQ(proposals->get_output_shape(0), (Shape{s.post_nms_count, 4}));
        EXPECT_EQ(proposals->get_output_shape(1), (Shape{s.post_nms_count}));
    }
}
