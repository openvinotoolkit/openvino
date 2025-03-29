// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/matrix_nms.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, matrix_nms_v8_op_custom_attributes) {
    NodeBuilder::opset().insert<ov::op::v8::MatrixNms>();
    auto boxes = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 1});

    ov::op::v8::MatrixNms::Attributes attrs;
    attrs.sort_result_type = ov::op::v8::MatrixNms::SortResultType::SCORE;
    attrs.output_type = ov::element::i32;
    attrs.nms_top_k = 100;
    attrs.keep_top_k = 10;
    attrs.sort_result_across_batch = true;
    attrs.score_threshold = 0.1f;
    attrs.background_class = 2;
    attrs.decay_function = ov::op::v8::MatrixNms::DecayFunction::GAUSSIAN;
    attrs.gaussian_sigma = 0.2f;
    attrs.post_threshold = 0.3f;
    attrs.normalized = false;

    auto nms = make_shared<ov::op::v8::MatrixNms>(boxes, scores, attrs);
    NodeBuilder builder(nms, {boxes, scores});
    auto g_nms = ov::as_type_ptr<ov::op::v8::MatrixNms>(builder.create());
    const auto expected_attr_count = 11;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    auto& g_nms_attrs = g_nms->get_attrs();
    auto& nms_attrs = nms->get_attrs();

    EXPECT_EQ(g_nms_attrs.sort_result_type, nms_attrs.sort_result_type);
    EXPECT_EQ(g_nms_attrs.output_type, nms_attrs.output_type);
    EXPECT_EQ(g_nms_attrs.nms_top_k, nms_attrs.nms_top_k);
    EXPECT_EQ(g_nms_attrs.keep_top_k, nms_attrs.keep_top_k);
    EXPECT_EQ(g_nms_attrs.sort_result_across_batch, nms_attrs.sort_result_across_batch);
    EXPECT_EQ(g_nms_attrs.score_threshold, nms_attrs.score_threshold);
    EXPECT_EQ(g_nms_attrs.background_class, nms_attrs.background_class);
    EXPECT_EQ(g_nms_attrs.decay_function, nms_attrs.decay_function);
    EXPECT_EQ(g_nms_attrs.gaussian_sigma, nms_attrs.gaussian_sigma);
    EXPECT_EQ(g_nms_attrs.post_threshold, nms_attrs.post_threshold);
    EXPECT_EQ(g_nms_attrs.normalized, nms_attrs.normalized);

    EXPECT_EQ(attrs.sort_result_type, nms_attrs.sort_result_type);
    EXPECT_EQ(attrs.output_type, nms_attrs.output_type);
    EXPECT_EQ(attrs.nms_top_k, nms_attrs.nms_top_k);
    EXPECT_EQ(attrs.keep_top_k, nms_attrs.keep_top_k);
    EXPECT_EQ(attrs.sort_result_across_batch, nms_attrs.sort_result_across_batch);
    EXPECT_EQ(attrs.score_threshold, nms_attrs.score_threshold);
    EXPECT_EQ(attrs.background_class, nms_attrs.background_class);
    EXPECT_EQ(attrs.decay_function, nms_attrs.decay_function);
    EXPECT_EQ(attrs.gaussian_sigma, nms_attrs.gaussian_sigma);
    EXPECT_EQ(attrs.post_threshold, nms_attrs.post_threshold);
    EXPECT_EQ(attrs.normalized, nms_attrs.normalized);
}

TEST(attributes, matrix_nms_v8_op_default_attributes) {
    NodeBuilder::opset().insert<ov::op::v8::MatrixNms>();
    auto boxes = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 1});

    auto nms = make_shared<ov::op::v8::MatrixNms>(boxes, scores, ov::op::v8::MatrixNms::Attributes());
    NodeBuilder builder(nms, {boxes, scores});
    auto g_nms = ov::as_type_ptr<ov::op::v8::MatrixNms>(builder.create());
    const auto expected_attr_count = 11;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    auto& g_nms_attrs = g_nms->get_attrs();
    auto& nms_attrs = nms->get_attrs();

    EXPECT_EQ(g_nms_attrs.sort_result_type, nms_attrs.sort_result_type);
    EXPECT_EQ(g_nms_attrs.output_type, nms_attrs.output_type);
    EXPECT_EQ(g_nms_attrs.nms_top_k, nms_attrs.nms_top_k);
    EXPECT_EQ(g_nms_attrs.keep_top_k, nms_attrs.keep_top_k);
    EXPECT_EQ(g_nms_attrs.sort_result_across_batch, nms_attrs.sort_result_across_batch);
    EXPECT_EQ(g_nms_attrs.score_threshold, nms_attrs.score_threshold);
    EXPECT_EQ(g_nms_attrs.background_class, nms_attrs.background_class);
    EXPECT_EQ(g_nms_attrs.decay_function, nms_attrs.decay_function);
    EXPECT_EQ(g_nms_attrs.gaussian_sigma, nms_attrs.gaussian_sigma);
    EXPECT_EQ(g_nms_attrs.post_threshold, nms_attrs.post_threshold);
    EXPECT_EQ(g_nms_attrs.normalized, nms_attrs.normalized);
}
