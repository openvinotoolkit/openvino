// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/multiclass_nms.hpp"

#include "gtest/gtest.h"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, multiclass_nms_v8_op_custom_attributes) {
    NodeBuilder::get_ops().register_factory<op::v8::MulticlassNms>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    op::v8::MulticlassNms::Attributes attrs;
    attrs.sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    attrs.sort_result_across_batch = true;
    attrs.output_type = ngraph::element::i32;
    attrs.nms_top_k = 100;
    attrs.keep_top_k = 10;
    attrs.iou_threshold = 0.1f;
    attrs.score_threshold = 0.2f;
    attrs.background_class = 2;
    attrs.nms_eta = 0.3f;
    attrs.normalized = false;

    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);
    NodeBuilder builder(nms);
    auto g_nms = as_type_ptr<op::v8::MulticlassNms>(builder.create());
    const auto expected_attr_count = 10;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    auto& g_nms_attrs = g_nms->get_attrs();
    auto& nms_attrs = nms->get_attrs();

    EXPECT_EQ(g_nms_attrs.sort_result_type, nms_attrs.sort_result_type);
    EXPECT_EQ(g_nms_attrs.sort_result_across_batch, nms_attrs.sort_result_across_batch);
    EXPECT_EQ(g_nms_attrs.output_type, nms_attrs.output_type);
    EXPECT_EQ(g_nms_attrs.nms_top_k, nms_attrs.nms_top_k);
    EXPECT_EQ(g_nms_attrs.keep_top_k, nms_attrs.keep_top_k);
    EXPECT_EQ(g_nms_attrs.iou_threshold, nms_attrs.iou_threshold);
    EXPECT_EQ(g_nms_attrs.score_threshold, nms_attrs.score_threshold);
    EXPECT_EQ(g_nms_attrs.background_class, nms_attrs.background_class);
    EXPECT_EQ(g_nms_attrs.nms_eta, nms_attrs.nms_eta);
    EXPECT_EQ(g_nms_attrs.normalized, nms_attrs.normalized);

    EXPECT_EQ(attrs.sort_result_type, nms_attrs.sort_result_type);
    EXPECT_EQ(attrs.sort_result_across_batch, nms_attrs.sort_result_across_batch);
    EXPECT_EQ(attrs.output_type, nms_attrs.output_type);
    EXPECT_EQ(attrs.nms_top_k, nms_attrs.nms_top_k);
    EXPECT_EQ(attrs.keep_top_k, nms_attrs.keep_top_k);
    EXPECT_EQ(attrs.iou_threshold, nms_attrs.iou_threshold);
    EXPECT_EQ(attrs.score_threshold, nms_attrs.score_threshold);
    EXPECT_EQ(attrs.background_class, nms_attrs.background_class);
    EXPECT_EQ(attrs.nms_eta, nms_attrs.nms_eta);
    EXPECT_EQ(attrs.normalized, nms_attrs.normalized);
}

TEST(attributes, multiclass_nms_v8_op_default_attributes) {
    NodeBuilder::get_ops().register_factory<op::v8::MulticlassNms>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, op::v8::MulticlassNms::Attributes());
    NodeBuilder builder(nms);
    auto g_nms = as_type_ptr<op::v8::MulticlassNms>(builder.create());
    const auto expected_attr_count = 10;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    auto& g_nms_attrs = g_nms->get_attrs();
    auto& nms_attrs = nms->get_attrs();

    EXPECT_EQ(g_nms_attrs.sort_result_type, nms_attrs.sort_result_type);
    EXPECT_EQ(g_nms_attrs.sort_result_across_batch, nms_attrs.sort_result_across_batch);
    EXPECT_EQ(g_nms_attrs.output_type, nms_attrs.output_type);
    EXPECT_EQ(g_nms_attrs.nms_top_k, nms_attrs.nms_top_k);
    EXPECT_EQ(g_nms_attrs.keep_top_k, nms_attrs.keep_top_k);
    EXPECT_EQ(g_nms_attrs.iou_threshold, nms_attrs.iou_threshold);
    EXPECT_EQ(g_nms_attrs.score_threshold, nms_attrs.score_threshold);
    EXPECT_EQ(g_nms_attrs.background_class, nms_attrs.background_class);
    EXPECT_EQ(g_nms_attrs.nms_eta, nms_attrs.nms_eta);
    EXPECT_EQ(g_nms_attrs.normalized, nms_attrs.normalized);
}
