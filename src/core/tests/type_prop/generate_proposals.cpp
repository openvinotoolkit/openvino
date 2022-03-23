// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;

using GenerateProposals = op::v9::GenerateProposalsSingleImage;
using Attrs = op::v9::GenerateProposalsSingleImage::Attributes;

TEST(type_prop, generate_proposals) {
    Attrs attrs;
    attrs.min_size = 0.0f;
    attrs.nms_threshold = 0.699999988079071f;
    attrs.post_nms_count = 1000;
    attrs.pre_nms_count = 1000;

    const auto dyn_dim = Dimension::dynamic();

    auto im_info = std::make_shared<op::Parameter>(element::f32, Shape{1, 4});
    auto anchors = std::make_shared<op::Parameter>(element::f32, Shape{200, 336, 3, 4});
    auto deltas = std::make_shared<op::Parameter>(element::f32, Shape{1, 12, 200, 336});
    auto scores = std::make_shared<op::Parameter>(element::f32, Shape{1, 3, 200, 336});

    auto proposals = std::make_shared<GenerateProposals>(im_info, anchors, deltas, scores, attrs);

    ASSERT_EQ(proposals->get_output_element_type(0), element::f32);
    ASSERT_EQ(proposals->get_output_element_type(1), element::f32);
    ASSERT_EQ(proposals->get_output_element_type(2), element::i64);
    EXPECT_EQ(proposals->get_output_partial_shape(0), (PartialShape{dyn_dim, 4}));
    EXPECT_EQ(proposals->get_output_partial_shape(1), (PartialShape{dyn_dim}));
    EXPECT_EQ(proposals->get_output_partial_shape(2), (PartialShape{1}));

    im_info = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic(2));
    anchors = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic(4));
    deltas = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic(4));
    scores = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic(4));

    proposals = std::make_shared<GenerateProposals>(im_info, anchors, deltas, scores, attrs, element::i32);

    ASSERT_EQ(proposals->get_output_element_type(0), element::f32);
    ASSERT_EQ(proposals->get_output_element_type(1), element::f32);
    ASSERT_EQ(proposals->get_output_element_type(2), element::i32);
    EXPECT_EQ(proposals->get_output_partial_shape(0), (PartialShape{dyn_dim, 4}));
    EXPECT_EQ(proposals->get_output_partial_shape(1), (PartialShape{dyn_dim}));
    EXPECT_EQ(proposals->get_output_partial_shape(2), (PartialShape{dyn_dim}));
}

TEST(type_prop, generate_proposals_dynamic) {
    struct ShapesAndAttrs {
        PartialShape im_info_shape;
        PartialShape anchors_shape;
        PartialShape deltas_shape;
        PartialShape scores_shape;
        size_t post_nms_count;
    };

    const auto dyn_dim = Dimension::dynamic();
    const auto dyn_shape = PartialShape::dynamic();

    std::vector<ShapesAndAttrs> shapes = {
        {{1, 3}, {200, 336, 3, 4}, {1, 12, 200, 336}, {1, 3, 200, 336}, 1000},
        {{2, 3}, {200, 336, 3, 4}, {2, 12, 200, 336}, dyn_shape, 500},
        {{1, 3}, {200, 336, 3, 4}, dyn_shape, {1, 3, 200, 336}, 700},
        {{2, 3}, {200, 336, 3, 4}, dyn_shape, dyn_shape, 300},
        {{1, 3}, dyn_shape, {1, 12, 200, 336}, {1, 3, 200, 336}, 200},
        {{2, 3}, dyn_shape, {2, 12, 200, 336}, dyn_shape, 40},
        {{1, 3}, dyn_shape, dyn_shape, {1, 3, 200, 336}, 70},
        {{2, 3}, dyn_shape, dyn_shape, dyn_shape, 60},
        {dyn_shape, {200, 336, 3, 4}, {1, 12, 200, 336}, {1, 3, 200, 336}, 500},
        {dyn_shape, {200, 336, 3, 4}, {2, 12, 200, 336}, dyn_shape, 400},
        {dyn_shape, {200, 336, 3, 4}, dyn_shape, {1, 3, 200, 336}, 350},
        {dyn_shape, {200, 336, 3, 4}, dyn_shape, dyn_shape, 440},
        {dyn_shape, dyn_shape, {1, 12, 200, 336}, {1, 3, 200, 336}, 315},
        {dyn_shape, dyn_shape, {2, 12, 200, 336}, dyn_shape, 130},
        {dyn_shape, dyn_shape, dyn_shape, {1, 3, 200, 336}, 1000},
        {dyn_shape, dyn_shape, dyn_shape, dyn_shape, 700},
        {{1, 3}, {dyn_dim, dyn_dim, dyn_dim, 4}, {1, 12, 200, 336}, {1, 3, 200, 336}, 540},
        {{1, 3}, {dyn_dim, dyn_dim, dyn_dim, 4}, {1, 12, 200, 336}, {dyn_dim, dyn_dim, 200, 336}, 600},
        {{2, 3}, {dyn_dim, dyn_dim, dyn_dim, 4}, {dyn_dim, dyn_dim, 200, 336}, {2, 3, 200, 336}, 75},
        {{1, 3}, {dyn_dim, dyn_dim, dyn_dim, 4}, {dyn_dim, dyn_dim, 200, 336}, {dyn_dim, dyn_dim, 200, 336}, 80},
        {{1, 3}, {200, 336, 3, 4}, {1, 12, 200, dyn_dim}, {1, 3, 200, dyn_dim}, 430},
        {{2, 3}, {200, 336, 3, 4}, {2, 12, dyn_dim, 336}, {2, 3, dyn_dim, 336}, 180},
        {{1, 3}, {200, 336, 3, 4}, {1, 12, dyn_dim, dyn_dim}, {1, 3, dyn_dim, dyn_dim}, 170},
        {{1, 3}, {dyn_dim, dyn_dim, dyn_dim, 4}, {1, 12, 200, dyn_dim}, {1, 3, 200, dyn_dim}, 200},
        {{2, 3}, {dyn_dim, dyn_dim, dyn_dim, 4}, {2, 12, dyn_dim, 336}, {2, 3, dyn_dim, 336}, 800},
        {{1, 3}, {dyn_dim, dyn_dim, dyn_dim, 4}, {1, 12, dyn_dim, dyn_dim}, {1, 3, dyn_dim, dyn_dim}, 560},
    };

    for (const auto& s : shapes) {
        Attrs attrs;
        attrs.min_size = 0.0f;
        attrs.nms_threshold = 0.699999988079071f;
        attrs.post_nms_count = static_cast<int64_t>(s.post_nms_count);
        attrs.pre_nms_count = 1000;

        auto im_info = std::make_shared<op::Parameter>(element::f32, s.im_info_shape);
        auto anchors = std::make_shared<op::Parameter>(element::f32, s.anchors_shape);
        auto deltas = std::make_shared<op::Parameter>(element::f32, s.deltas_shape);
        auto scores = std::make_shared<op::Parameter>(element::f32, s.scores_shape);

        auto proposals = std::make_shared<GenerateProposals>(im_info, anchors, deltas, scores, attrs);

        ASSERT_EQ(proposals->get_output_element_type(0), element::f32);
        ASSERT_EQ(proposals->get_output_element_type(1), element::f32);
        ASSERT_EQ(proposals->get_output_element_type(2), element::i64);
        EXPECT_EQ(proposals->get_output_partial_shape(0), (PartialShape{dyn_dim, 4}));
        EXPECT_EQ(proposals->get_output_partial_shape(1), (PartialShape{dyn_dim}));
        if (s.im_info_shape.rank().is_static()) {
            EXPECT_EQ(proposals->get_output_partial_shape(2), PartialShape{s.im_info_shape[0]});
        } else {
            EXPECT_EQ(proposals->get_output_partial_shape(2), PartialShape::dynamic());
        }
    }
}
