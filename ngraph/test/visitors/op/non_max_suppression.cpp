// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/op/non_max_suppression.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, non_max_suppression_op_custom_attributes) {
    NodeBuilder::get_ops().register_factory<op::v1::NonMaxSuppression>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto box_encoding = op::v1::NonMaxSuppression::BoxEncodingType::CENTER;
    bool sort_result_descending = false;

    auto nms = make_shared<op::v1::NonMaxSuppression>(boxes, scores, box_encoding, sort_result_descending);
    NodeBuilder builder(nms);
    auto g_nms = as_type_ptr<op::v1::NonMaxSuppression>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
}

TEST(attributes, non_max_suppression_op_default_attributes) {
    NodeBuilder::get_ops().register_factory<op::v1::NonMaxSuppression>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto nms = make_shared<op::v1::NonMaxSuppression>(boxes, scores);
    NodeBuilder builder(nms);
    auto g_nms = as_type_ptr<op::v1::NonMaxSuppression>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
}

TEST(attributes, non_max_suppression_v3_op_custom_attributes) {
    NodeBuilder::get_ops().register_factory<op::v3::NonMaxSuppression>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto box_encoding = op::v3::NonMaxSuppression::BoxEncodingType::CENTER;
    bool sort_result_descending = false;
    element::Type output_type = element::i32;

    auto nms = make_shared<op::v3::NonMaxSuppression>(boxes, scores, box_encoding, sort_result_descending, output_type);
    NodeBuilder builder(nms);
    auto g_nms = as_type_ptr<op::v3::NonMaxSuppression>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
    EXPECT_EQ(g_nms->get_output_type(), nms->get_output_type());
}

TEST(attributes, non_max_suppression_v3_op_default_attributes) {
    NodeBuilder::get_ops().register_factory<op::v3::NonMaxSuppression>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto nms = make_shared<op::v3::NonMaxSuppression>(boxes, scores);
    NodeBuilder builder(nms);
    auto g_nms = as_type_ptr<op::v3::NonMaxSuppression>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
    EXPECT_EQ(g_nms->get_output_type(), nms->get_output_type());
}
