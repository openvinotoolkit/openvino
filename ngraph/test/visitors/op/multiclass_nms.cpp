// Copyright (C) 2018-2021 Intel Corporation
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

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, multiclass_nms_v8_op_custom_attributes)
{
    NodeBuilder::get_ops().register_factory<opset8::MulticlassNms>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto sort_result_type = opset8::MulticlassNms::SortResultType::SCORE;
    auto sort_result_across_batch = false;
    auto output_type = ngraph::element::i32;
    int nms_top_k = 100;
    int keep_top_k = 10;
    float iou_threshold = 0.1f;
    float score_threshold = 0.2f;
    int background_class = 2;
    float nms_eta = 0.3f;

    auto nms = make_shared<opset8::MulticlassNms>(boxes, scores, sort_result_type, sort_result_across_batch, output_type, iou_threshold, score_threshold, nms_top_k, keep_top_k, background_class, nms_eta);
    NodeBuilder builder(nms);
    auto g_nms = as_type_ptr<opset8::MulticlassNms>(builder.create());

    EXPECT_EQ(g_nms->get_sort_result_type(), nms->get_sort_result_type());
    EXPECT_EQ(g_nms->get_sort_result_across_batch(), nms->get_sort_result_across_batch());
    EXPECT_EQ(g_nms->get_output_type(), nms->get_output_type());
    EXPECT_EQ(g_nms->get_nms_top_k(), nms->get_nms_top_k());
    EXPECT_EQ(g_nms->get_keep_top_k(), nms->get_keep_top_k());
    EXPECT_EQ(g_nms->get_iou_threshold(), nms->get_iou_threshold());
    EXPECT_EQ(g_nms->get_score_threshold(), nms->get_score_threshold());
    EXPECT_EQ(g_nms->get_background_class(), nms->get_background_class());
    EXPECT_EQ(g_nms->get_nms_eta(), nms->get_nms_eta());
    EXPECT_EQ(sort_result_type, nms->get_sort_result_type());
    EXPECT_EQ(sort_result_across_batch, nms->get_sort_result_across_batch());
    EXPECT_EQ(output_type, nms->get_output_type());
    EXPECT_EQ(nms_top_k, nms->get_nms_top_k());
    EXPECT_EQ(keep_top_k, nms->get_keep_top_k());
    EXPECT_EQ(iou_threshold, nms->get_iou_threshold());
    EXPECT_EQ(score_threshold, nms->get_score_threshold());
    EXPECT_EQ(background_class, nms->get_background_class());
    EXPECT_EQ(nms_eta, nms->get_nms_eta());    
}

TEST(attributes, multiclass_nms_v8_op_default_attributes)
{
    NodeBuilder::get_ops().register_factory<opset8::MulticlassNms>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto nms = make_shared<opset8::MulticlassNms>(boxes, scores);
    NodeBuilder builder(nms);
    auto g_nms = as_type_ptr<opset8::MulticlassNms>(builder.create());

    EXPECT_EQ(g_nms->get_sort_result_type(), nms->get_sort_result_type());
    EXPECT_EQ(g_nms->get_sort_result_across_batch(), nms->get_sort_result_across_batch());
    EXPECT_EQ(g_nms->get_output_type(), nms->get_output_type());
    EXPECT_EQ(g_nms->get_nms_top_k(), nms->get_nms_top_k());
    EXPECT_EQ(g_nms->get_keep_top_k(), nms->get_keep_top_k());
    EXPECT_EQ(g_nms->get_iou_threshold(), nms->get_iou_threshold());
    EXPECT_EQ(g_nms->get_score_threshold(), nms->get_score_threshold());
    EXPECT_EQ(g_nms->get_background_class(), nms->get_background_class());
    EXPECT_EQ(g_nms->get_nms_eta(), nms->get_nms_eta());
}
