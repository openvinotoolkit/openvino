// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, non_max_suppression_op_custom_attributes) {
    NodeBuilder::get_ops().register_factory<opset1::NonMaxSuppression>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto box_encoding = opset1::NonMaxSuppression::BoxEncodingType::CENTER;
    bool sort_result_descending = false;

    auto nms = make_shared<opset1::NonMaxSuppression>(boxes, scores, box_encoding, sort_result_descending);
    NodeBuilder builder(nms);
    auto g_nms = ov::as_type_ptr<opset1::NonMaxSuppression>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
}

TEST(attributes, non_max_suppression_op_default_attributes) {
    NodeBuilder::get_ops().register_factory<opset1::NonMaxSuppression>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto nms = make_shared<opset1::NonMaxSuppression>(boxes, scores);
    NodeBuilder builder(nms);
    auto g_nms = ov::as_type_ptr<opset1::NonMaxSuppression>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
}

TEST(attributes, non_max_suppression_v3_op_custom_attributes) {
    NodeBuilder::get_ops().register_factory<opset3::NonMaxSuppression>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto box_encoding = opset3::NonMaxSuppression::BoxEncodingType::CENTER;
    bool sort_result_descending = false;
    element::Type output_type = element::i32;

    auto nms = make_shared<opset3::NonMaxSuppression>(boxes, scores, box_encoding, sort_result_descending, output_type);
    NodeBuilder builder(nms);
    auto g_nms = ov::as_type_ptr<opset3::NonMaxSuppression>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
    EXPECT_EQ(g_nms->get_output_type(), nms->get_output_type());
}

TEST(attributes, non_max_suppression_v3_op_default_attributes) {
    NodeBuilder::get_ops().register_factory<opset3::NonMaxSuppression>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto nms = make_shared<opset3::NonMaxSuppression>(boxes, scores);
    NodeBuilder builder(nms);
    auto g_nms = ov::as_type_ptr<opset3::NonMaxSuppression>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
    EXPECT_EQ(g_nms->get_output_type(), nms->get_output_type());
}
