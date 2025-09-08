// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/non_max_suppression.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, non_max_suppression_op_custom_attributes) {
    NodeBuilder::opset().insert<ov::op::v1::NonMaxSuppression>();
    auto boxes = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 1});

    auto box_encoding = ov::op::v1::NonMaxSuppression::BoxEncodingType::CENTER;
    bool sort_result_descending = false;

    auto nms = make_shared<ov::op::v1::NonMaxSuppression>(boxes, scores, box_encoding, sort_result_descending);
    NodeBuilder builder(nms, {boxes, scores});
    auto g_nms = ov::as_type_ptr<ov::op::v1::NonMaxSuppression>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
}

TEST(attributes, non_max_suppression_op_default_attributes) {
    NodeBuilder::opset().insert<ov::op::v1::NonMaxSuppression>();
    auto boxes = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 1});

    auto nms = make_shared<ov::op::v1::NonMaxSuppression>(boxes, scores);
    NodeBuilder builder(nms, {boxes, scores});
    auto g_nms = ov::as_type_ptr<ov::op::v1::NonMaxSuppression>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
}

TEST(attributes, non_max_suppression_v3_op_custom_attributes) {
    NodeBuilder::opset().insert<ov::op::v3::NonMaxSuppression>();
    auto boxes = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 1});

    auto box_encoding = ov::op::v3::NonMaxSuppression::BoxEncodingType::CENTER;
    bool sort_result_descending = false;
    element::Type output_type = element::i32;

    auto nms =
        make_shared<ov::op::v3::NonMaxSuppression>(boxes, scores, box_encoding, sort_result_descending, output_type);
    NodeBuilder builder(nms, {boxes, scores});
    auto g_nms = ov::as_type_ptr<ov::op::v3::NonMaxSuppression>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
    EXPECT_EQ(g_nms->get_output_type(), nms->get_output_type());
}

TEST(attributes, non_max_suppression_v3_op_default_attributes) {
    NodeBuilder::opset().insert<ov::op::v3::NonMaxSuppression>();
    auto boxes = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 1});

    auto nms = make_shared<ov::op::v3::NonMaxSuppression>(boxes, scores);
    NodeBuilder builder(nms, {boxes, scores});
    auto g_nms = ov::as_type_ptr<ov::op::v3::NonMaxSuppression>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
    EXPECT_EQ(g_nms->get_output_type(), nms->get_output_type());
}
