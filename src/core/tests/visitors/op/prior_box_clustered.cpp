// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/prior_box_clustered.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, prior_box_clustered_op) {
    NodeBuilder::opset().insert<ov::op::v0::PriorBoxClustered>();
    const auto layer_shape = make_shared<ov::op::v0::Parameter>(element::i64, Shape{2});
    const auto image_shape = make_shared<ov::op::v0::Parameter>(element::i64, Shape{2});

    ov::op::v0::PriorBoxClustered::Attributes attrs;
    attrs.heights = {2.0f, 3.0f};
    attrs.widths = {2.0f, 3.0f};
    attrs.clip = true;
    attrs.step_widths = 0.0f;
    attrs.step_heights = 0.0f;
    attrs.step = 16.0f;
    attrs.offset = 0.0f;
    attrs.variances = {0.1f};

    auto pbc = make_shared<ov::op::v0::PriorBoxClustered>(layer_shape, image_shape, attrs);
    NodeBuilder builder(pbc, {layer_shape, image_shape});
    auto g_pbc = ov::as_type_ptr<ov::op::v0::PriorBoxClustered>(builder.create());
    const auto pbc_attrs = pbc->get_attrs();
    const auto g_pbc_attrs = g_pbc->get_attrs();
    const auto expected_attr_count = 8;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_pbc_attrs.heights, pbc_attrs.heights);
    EXPECT_EQ(g_pbc_attrs.widths, pbc_attrs.widths);
    EXPECT_EQ(g_pbc_attrs.clip, pbc_attrs.clip);
    EXPECT_EQ(g_pbc_attrs.step_widths, pbc_attrs.step_widths);
    EXPECT_EQ(g_pbc_attrs.step_heights, pbc_attrs.step_heights);
    EXPECT_EQ(g_pbc_attrs.step, pbc_attrs.step);
    EXPECT_EQ(g_pbc_attrs.offset, pbc_attrs.offset);
    EXPECT_EQ(g_pbc_attrs.variances, pbc_attrs.variances);
    EXPECT_EQ(g_pbc->has_evaluate(), pbc->has_evaluate());
}

TEST(attributes, prior_box_clustered_op2) {
    NodeBuilder::opset().insert<ov::op::v0::PriorBoxClustered>();
    const auto layer_shape = make_shared<ov::op::v0::Parameter>(element::i64, Shape{2});
    const auto image_shape = make_shared<ov::op::v0::Parameter>(element::i64, Shape{2});

    ov::op::v0::PriorBoxClustered::Attributes attrs;
    attrs.heights = {44.0f, 10.0f, 30.0f, 19.0f, 94.0f, 32.0f, 61.0f, 53.0f, 17.0f};
    attrs.widths = {86.0f, 13.0f, 57.0f, 39.0f, 68.0f, 34.0f, 142.0f, 50.0f, 23.0f};
    attrs.clip = false;
    attrs.step_widths = 0.0f;
    attrs.step_heights = 0.0f;
    attrs.step = 16.0f;
    attrs.offset = 0.5f;
    attrs.variances = {0.1f, 0.1f, 0.2f, 0.2f};

    auto pbc = make_shared<ov::op::v0::PriorBoxClustered>(layer_shape, image_shape, attrs);
    NodeBuilder builder(pbc, {layer_shape, image_shape});
    auto g_pbc = ov::as_type_ptr<ov::op::v0::PriorBoxClustered>(builder.create());
    const auto pbc_attrs = pbc->get_attrs();
    const auto g_pbc_attrs = g_pbc->get_attrs();
    const auto expected_attr_count = 8;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_pbc_attrs.heights, pbc_attrs.heights);
    EXPECT_EQ(g_pbc_attrs.widths, pbc_attrs.widths);
    EXPECT_EQ(g_pbc_attrs.clip, pbc_attrs.clip);
    EXPECT_EQ(g_pbc_attrs.step_widths, pbc_attrs.step_widths);
    EXPECT_EQ(g_pbc_attrs.step_heights, pbc_attrs.step_heights);
    EXPECT_EQ(g_pbc_attrs.step, pbc_attrs.step);
    EXPECT_EQ(g_pbc_attrs.offset, pbc_attrs.offset);
    EXPECT_EQ(g_pbc_attrs.variances, pbc_attrs.variances);
    EXPECT_EQ(g_pbc->has_evaluate(), pbc->has_evaluate());
}
