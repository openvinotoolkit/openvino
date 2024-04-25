// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/generate_proposals.hpp"

#include <gtest/gtest.h>

#include <vector>

#include "common_test_utils/type_prop.hpp"

using namespace ov;
using namespace testing;

using GenerateProposals = op::v9::GenerateProposals;
using Attrs = op::v9::GenerateProposals::Attributes;

TEST(type_prop, generate_proposals_default_ctor) {
    struct ShapesAndAttrs {
        PartialShape im_info_shape;
        PartialShape anchors_shape;
        PartialShape deltas_shape;
        PartialShape scores_shape;
        size_t post_nms_count;
        PartialShape expected_shape_0;
        PartialShape expected_shape_1;
        PartialShape expected_shape_2;
    };

    const auto dyn_dim = Dimension::dynamic();
    const auto dyn_shape = PartialShape::dynamic();

    ShapesAndAttrs s =
        {{1, 3}, {200, 336, 3, 4}, {1, 12, 200, 336}, {1, 3, 200, 336}, 1000, {{0, 1000}, 4}, {{0, 1000}}, {1}};
    Attrs attrs;
    attrs.min_size = 0.0f;
    attrs.nms_threshold = 0.699999988079071f;
    attrs.post_nms_count = static_cast<int64_t>(s.post_nms_count);
    attrs.pre_nms_count = 1000;

    auto im_info = std::make_shared<ov::op::v0::Parameter>(element::f32, s.im_info_shape);
    auto anchors = std::make_shared<ov::op::v0::Parameter>(element::f32, s.anchors_shape);
    auto deltas = std::make_shared<ov::op::v0::Parameter>(element::f32, s.deltas_shape);
    auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, s.scores_shape);

    auto proposals = std::make_shared<GenerateProposals>();
    proposals->set_arguments(OutputVector{im_info, anchors, deltas, scores});
    proposals->set_attrs(attrs);
    proposals->validate_and_infer_types();

    EXPECT_EQ(proposals->get_output_size(), 3);
    EXPECT_EQ(proposals->get_output_element_type(0), element::f32);
    EXPECT_EQ(proposals->get_output_element_type(1), element::f32);
    EXPECT_EQ(proposals->get_output_element_type(2), element::i64);
    EXPECT_EQ(proposals->get_output_partial_shape(0), s.expected_shape_0);
    EXPECT_EQ(proposals->get_output_partial_shape(1), s.expected_shape_1);
    EXPECT_EQ(proposals->get_output_partial_shape(2), s.expected_shape_2);
}

TEST(type_prop, generate_proposals) {
    Attrs attrs;
    attrs.min_size = 0.0f;
    attrs.nms_threshold = 0.699999988079071f;
    attrs.post_nms_count = 1000;
    attrs.pre_nms_count = 1000;

    const auto dyn_dim = Dimension::dynamic();

    auto im_info = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4});
    auto anchors = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{200, 336, 3, 4});
    auto deltas = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 12, 200, 336});
    auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 200, 336});

    auto proposals = std::make_shared<GenerateProposals>(im_info, anchors, deltas, scores, attrs);

    ASSERT_EQ(proposals->get_output_element_type(0), element::f32);
    ASSERT_EQ(proposals->get_output_element_type(1), element::f32);
    ASSERT_EQ(proposals->get_output_element_type(2), element::i64);
    EXPECT_EQ(proposals->get_output_partial_shape(0), (PartialShape{{0, 1000}, 4}));
    EXPECT_EQ(proposals->get_output_partial_shape(1), (PartialShape{{0, 1000}}));
    EXPECT_EQ(proposals->get_output_partial_shape(2), (PartialShape{1}));

    im_info = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    anchors = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    deltas = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    scores = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(4));

    proposals = std::make_shared<GenerateProposals>(im_info, anchors, deltas, scores, attrs, element::i32);

    ASSERT_EQ(proposals->get_output_element_type(0), element::f32);
    ASSERT_EQ(proposals->get_output_element_type(1), element::f32);
    ASSERT_EQ(proposals->get_output_element_type(2), element::i32);
    EXPECT_EQ(proposals->get_output_partial_shape(0), (PartialShape{dyn_dim, 4}));
    EXPECT_EQ(proposals->get_output_partial_shape(1), (PartialShape{dyn_dim}));
    EXPECT_EQ(proposals->get_output_partial_shape(2), (PartialShape{dyn_dim}));

    // assert throw
    im_info = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4});
    anchors = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{100, 336, 3, 4});
    deltas = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 12, 200, 336});
    scores = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 200, 336});

    ASSERT_THROW(proposals = std::make_shared<GenerateProposals>(im_info, anchors, deltas, scores, attrs, element::i32),
                 ov::AssertFailure)
        << "GenerateProposals node was created with invalid data.";

    im_info = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4});
    anchors = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{200, 336, 3, 4});
    deltas = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 12, 200, 300});
    scores = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 200, 336});

    ASSERT_THROW(proposals = std::make_shared<GenerateProposals>(im_info, anchors, deltas, scores, attrs, element::i32),
                 ov::AssertFailure)
        << "GenerateProposals node was created with invalid data.";

    im_info = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4});
    anchors = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{200, 336, 3, 4});
    deltas = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 12, 200, 336});
    scores = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 200, 336});

    ASSERT_THROW(proposals = std::make_shared<GenerateProposals>(im_info, anchors, deltas, scores, attrs, element::i32),
                 ov::AssertFailure)
        << "GenerateProposals node was created with invalid data.";

    im_info = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2});
    anchors = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{200, 336, 3, 4});
    deltas = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 12, 200, 336});
    scores = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 200, 336});

    ASSERT_THROW(proposals = std::make_shared<GenerateProposals>(im_info, anchors, deltas, scores, attrs, element::i32),
                 ov::AssertFailure)
        << "GenerateProposals node was created with invalid data.";

    im_info = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    anchors = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{200, 336, 3, 4});
    deltas = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 12, 200, 336});
    scores = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 200, 336});

    ASSERT_THROW(proposals = std::make_shared<GenerateProposals>(im_info, anchors, deltas, scores, attrs, element::i32),
                 ov::AssertFailure)
        << "GenerateProposals node was created with invalid data.";
}

TEST(type_prop, generate_proposals_dynamic) {
    struct ShapesAndAttrs {
        PartialShape im_info_shape;
        PartialShape anchors_shape;
        PartialShape deltas_shape;
        PartialShape scores_shape;
        size_t post_nms_count;
        PartialShape expected_shape_0;
        PartialShape expected_shape_1;
        PartialShape expected_shape_2;
    };

    const auto dyn_dim = Dimension::dynamic();
    const auto dyn_shape = PartialShape::dynamic();

    std::vector<ShapesAndAttrs> shapes = {
        {{1, 3}, {200, 336, 3, 4}, {1, 12, 200, 336}, {1, 3, 200, 336}, 1000, {{0, 1000}, 4}, {{0, 1000}}, {1}},
        {{{2, 4}, 3},
         {200, 336, 3, 4},
         {{2, 4}, 12, 200, 336},
         {{2, 4}, 3, 200, 336},
         1000,
         {{0, 4000}, 4},
         {{0, 4000}},
         {{2, 4}}},
        {{{2, 8}, 3},
         {200, 336, 3, 4},
         {{2, 6}, 12, 200, 336},
         {{6, 8}, 3, 200, 336},
         1000,
         {{0, 6000}, 4},
         {{0, 6000}},
         {{6}}},
        {{dyn_dim, 3},
         {200, 336, 3, 4},
         {dyn_dim, 12, 200, 336},
         {dyn_dim, 3, 200, 336},
         1000,
         {dyn_dim, 4},
         {dyn_dim},
         {dyn_dim}},
        {{2, 3}, {200, 336, 3, 4}, {2, 12, 200, 336}, dyn_shape, 500, {{0, 1000}, 4}, {{0, 1000}}, {2}},
        {{1, 3}, {200, 336, 3, 4}, dyn_shape, {1, 3, 200, 336}, 700, {{0, 700}, 4}, {{0, 700}}, {1}},
        {{2, 3}, {200, 336, 3, 4}, dyn_shape, dyn_shape, 300, {{0, 600}, 4}, {{0, 600}}, {2}},
        {{1, 3}, dyn_shape, {1, 12, 200, 336}, {1, 3, 200, 336}, 200, {{0, 200}, 4}, {{0, 200}}, {1}},
        {{2, 3}, dyn_shape, {2, 12, 200, 336}, dyn_shape, 40, {{0, 80}, 4}, {{0, 80}}, {2}},
        {{1, 3}, dyn_shape, dyn_shape, {1, 3, 200, 336}, 70, {{0, 70}, 4}, {{0, 70}}, {1}},
        {{2, 3}, dyn_shape, dyn_shape, dyn_shape, 60, {{0, 120}, 4}, {{0, 120}}, {2}},
        {dyn_shape, {200, 336, 3, 4}, {1, 12, 200, 336}, {1, 3, 200, 336}, 500, {{0, 500}, 4}, {{0, 500}}, {1}},
        {dyn_shape, {200, 336, 3, 4}, {2, 12, 200, 336}, dyn_shape, 400, {{0, 800}, 4}, {{0, 800}}, {2}},
        {dyn_shape, {200, 336, 3, 4}, dyn_shape, {1, 3, 200, 336}, 350, {{0, 350}, 4}, {{0, 350}}, {1}},
        {dyn_shape, {200, 336, 3, 4}, dyn_shape, dyn_shape, 440, {dyn_dim, 4}, {dyn_dim}, {dyn_dim}},
        {dyn_shape, dyn_shape, {1, 12, 200, 336}, {1, 3, 200, 336}, 315, {{0, 315}, 4}, {{0, 315}}, {1}},
        {dyn_shape, dyn_shape, {2, 12, 200, 336}, dyn_shape, 130, {{0, 260}, 4}, {{0, 260}}, {2}},
        {dyn_shape, dyn_shape, dyn_shape, {1, 3, 200, 336}, 1000, {{0, 1000}, 4}, {{0, 1000}}, {1}},
        {dyn_shape, dyn_shape, dyn_shape, dyn_shape, 700, {dyn_dim, 4}, {dyn_dim}, {dyn_dim}},
        {{1, 3},
         {dyn_dim, dyn_dim, dyn_dim, 4},
         {1, 12, 200, 336},
         {1, 3, 200, 336},
         540,
         {{0, 540}, 4},
         {{0, 540}},
         {{1}}},
        {{1, 3},
         {dyn_dim, dyn_dim, dyn_dim, 4},
         {1, 12, 200, 336},
         {dyn_dim, dyn_dim, 200, 336},
         600,
         {{0, 600}, 4},
         {{0, 600}},
         {{1}}},
        {{2, 3},
         {dyn_dim, dyn_dim, dyn_dim, 4},
         {dyn_dim, dyn_dim, 200, 336},
         {2, 3, 200, 336},
         75,
         {{0, 150}, 4},
         {{0, 150}},
         {{2}}},
        {{1, 3},
         {dyn_dim, dyn_dim, dyn_dim, 4},
         {dyn_dim, dyn_dim, 200, 336},
         {dyn_dim, dyn_dim, 200, 336},
         80,
         {{0, 80}, 4},
         {{0, 80}},
         {{1}}},
        {{1, 3}, {200, 336, 3, 4}, {1, 12, 200, dyn_dim}, {1, 3, 200, dyn_dim}, 430, {{0, 430}, 4}, {{0, 430}}, {{1}}},
        {{2, 3}, {200, 336, 3, 4}, {2, 12, dyn_dim, 336}, {2, 3, dyn_dim, 336}, 180, {{0, 360}, 4}, {{0, 360}}, {{2}}},
        {{1, 3},
         {200, 336, 3, 4},
         {1, 12, dyn_dim, dyn_dim},
         {1, 3, dyn_dim, dyn_dim},
         170,
         {{0, 170}, 4},
         {{0, 170}},
         {{1}}},
        {{1, 3},
         {dyn_dim, dyn_dim, dyn_dim, 4},
         {1, 12, 200, dyn_dim},
         {1, 3, 200, dyn_dim},
         200,
         {{0, 200}, 4},
         {{0, 200}},
         {{1}}},
        {{2, 3},
         {dyn_dim, dyn_dim, dyn_dim, 4},
         {2, 12, dyn_dim, 336},
         {2, 3, dyn_dim, 336},
         800,
         {{0, 1600}, 4},
         {{0, 1600}},
         {{2}}},
        {{1, 3},
         {dyn_dim, dyn_dim, dyn_dim, 4},
         {1, 12, dyn_dim, dyn_dim},
         {1, 3, dyn_dim, dyn_dim},
         560,
         {{0, 560}, 4},
         {{0, 560}},
         {{1}}},
    };

    for (auto& s : shapes) {
        Attrs attrs;
        attrs.min_size = 0.0f;
        attrs.nms_threshold = 0.699999988079071f;
        attrs.post_nms_count = static_cast<int64_t>(s.post_nms_count);
        attrs.pre_nms_count = 1000;

        if (s.im_info_shape.rank().is_static())
            set_shape_symbols(s.im_info_shape);
        if (s.anchors_shape.rank().is_static())
            set_shape_symbols(s.anchors_shape);
        if (s.deltas_shape.rank().is_static())
            set_shape_symbols(s.deltas_shape);
        if (s.scores_shape.rank().is_static())
            set_shape_symbols(s.scores_shape);

        auto expected_batch_label = s.scores_shape.rank().is_static()    ? s.scores_shape[0].get_symbol()
                                    : s.deltas_shape.rank().is_static()  ? s.deltas_shape[0].get_symbol()
                                    : s.im_info_shape.rank().is_static() ? s.im_info_shape[0].get_symbol()
                                                                         : nullptr;

        auto im_info = std::make_shared<ov::op::v0::Parameter>(element::f32, s.im_info_shape);
        auto anchors = std::make_shared<ov::op::v0::Parameter>(element::f32, s.anchors_shape);
        auto deltas = std::make_shared<ov::op::v0::Parameter>(element::f32, s.deltas_shape);
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, s.scores_shape);

        auto proposals = std::make_shared<GenerateProposals>(im_info, anchors, deltas, scores, attrs);

        EXPECT_EQ(proposals->get_output_size(), 3);
        EXPECT_EQ(proposals->get_output_element_type(0), element::f32);
        EXPECT_EQ(proposals->get_output_element_type(1), element::f32);
        EXPECT_EQ(proposals->get_output_element_type(2), element::i64);
        EXPECT_EQ(proposals->get_output_partial_shape(0), s.expected_shape_0);
        EXPECT_EQ(proposals->get_output_partial_shape(1), s.expected_shape_1);
        EXPECT_EQ(proposals->get_output_partial_shape(2), s.expected_shape_2);
        EXPECT_THAT(get_shape_symbols(proposals->get_output_partial_shape(0)), ElementsAre(nullptr, nullptr));
        EXPECT_THAT(get_shape_symbols(proposals->get_output_partial_shape(1)), ElementsAre(nullptr));
        EXPECT_THAT(get_shape_symbols(proposals->get_output_partial_shape(2)), ElementsAre(expected_batch_label));
    }
}
