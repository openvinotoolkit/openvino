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
#include "ngraph/opsets/opset8.hpp"
#include "ngraph/opsets/opset9.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, multiclass_nms_v8_op_custom_attributes) {
    NodeBuilder::get_ops().register_factory<opset8::MulticlassNms>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    opset8::MulticlassNms::Attributes attrs;
    attrs.sort_result_type = opset8::MulticlassNms::SortResultType::SCORE;
    attrs.sort_result_across_batch = true;
    attrs.output_type = ngraph::element::i32;
    attrs.nms_top_k = 100;
    attrs.keep_top_k = 10;
    attrs.iou_threshold = 0.1f;
    attrs.score_threshold = 0.2f;
    attrs.background_class = 2;
    attrs.nms_eta = 0.3f;
    attrs.normalized = false;

    auto nms = make_shared<opset8::MulticlassNms>(boxes, scores, attrs);
    NodeBuilder builder(nms);
    auto g_nms = ov::as_type_ptr<opset8::MulticlassNms>(builder.create());
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
    NodeBuilder::get_ops().register_factory<opset8::MulticlassNms>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto nms = make_shared<opset8::MulticlassNms>(boxes, scores, opset8::MulticlassNms::Attributes());
    NodeBuilder builder(nms);
    auto g_nms = ov::as_type_ptr<opset8::MulticlassNms>(builder.create());
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

TEST(attributes, multiclass_nms_v9_op_custom_attributes) {
    NodeBuilder::get_ops().register_factory<opset9::MulticlassNms>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{3, 2, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{3, 2});
    auto roisnum = make_shared<op::Parameter>(element::f32, Shape{2});

    opset9::MulticlassNms::Attributes attrs;
    attrs.sort_result_type = opset9::MulticlassNms::SortResultType::SCORE;
    attrs.sort_result_across_batch = true;
    attrs.output_type = ngraph::element::i32;
    attrs.nms_top_k = 100;
    attrs.keep_top_k = 10;
    attrs.iou_threshold = 0.1f;
    attrs.score_threshold = 0.2f;
    attrs.background_class = 2;
    attrs.nms_eta = 0.3f;
    attrs.normalized = false;

    auto nms = make_shared<opset9::MulticlassNms>(boxes, scores, roisnum, attrs);
    NodeBuilder builder(nms);
    auto g_nms = ov::as_type_ptr<opset9::MulticlassNms>(builder.create());
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

TEST(attributes, multiclass_nms_v9_op_default_attributes) {
    NodeBuilder::get_ops().register_factory<opset9::MulticlassNms>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{3, 2, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{3, 2});
    auto roisnum = make_shared<op::Parameter>(element::f32, Shape{2});

    auto nms = make_shared<opset9::MulticlassNms>(boxes, scores, roisnum, opset9::MulticlassNms::Attributes());
    NodeBuilder builder(nms);
    auto g_nms = ov::as_type_ptr<opset9::MulticlassNms>(builder.create());
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