// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental_detectron_generate_proposals.hpp"

#include <gtest/gtest.h>

#include <vector>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

using ExperimentalProposals = ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage;
using Attrs = ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage::Attributes;

TEST(attributes, detectron_proposals) {
    NodeBuilder::opset().insert<ExperimentalProposals>();

    Attrs attrs;
    attrs.min_size = 0.0f;
    attrs.nms_threshold = 0.699999988079071f;
    attrs.post_nms_count = 1000;
    attrs.pre_nms_count = 1000;

    auto im_info = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});
    auto anchors = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{201600, 4});
    auto deltas = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{12, 200, 336});
    auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 200, 336});

    auto proposals = std::make_shared<ExperimentalProposals>(im_info, anchors, deltas, scores, attrs);

    NodeBuilder builder(proposals, {im_info, anchors, deltas, scores});

    auto g_proposals = ov::as_type_ptr<ExperimentalProposals>(builder.create());

    const auto expected_attr_count = 4;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_proposals->get_attrs().min_size, proposals->get_attrs().min_size);
    EXPECT_EQ(g_proposals->get_attrs().nms_threshold, proposals->get_attrs().nms_threshold);
    EXPECT_EQ(g_proposals->get_attrs().post_nms_count, proposals->get_attrs().post_nms_count);
    EXPECT_EQ(g_proposals->get_attrs().pre_nms_count, proposals->get_attrs().pre_nms_count);
}
