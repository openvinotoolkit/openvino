// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset9.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

using GenerateProposals = opset9::GenerateProposals;
using Attrs = opset9::GenerateProposals::Attributes;

TEST(attributes, generate_proposals) {
    NodeBuilder::get_ops().register_factory<GenerateProposals>();

    Attrs attrs;
    attrs.min_size = 0.0f;
    attrs.nms_threshold = 0.699999988079071f;
    attrs.post_nms_count = 1000;
    attrs.pre_nms_count = 1000;
    attrs.normalized = true;
    attrs.nms_eta = 1.0f;

    auto im_info = std::make_shared<op::Parameter>(element::f32, Shape{1, 4});
    auto anchors = std::make_shared<op::Parameter>(element::f32, Shape{200, 336, 3, 4});
    auto deltas = std::make_shared<op::Parameter>(element::f32, Shape{1, 12, 200, 336});
    auto scores = std::make_shared<op::Parameter>(element::f32, Shape{1, 3, 200, 336});

    auto proposals = std::make_shared<GenerateProposals>(im_info, anchors, deltas, scores, attrs);

    NodeBuilder builder(proposals);

    auto g_proposals = ov::as_type_ptr<GenerateProposals>(builder.create());

    const auto expected_attr_count = 7;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_proposals->get_attrs().min_size, proposals->get_attrs().min_size);
    EXPECT_EQ(g_proposals->get_attrs().nms_threshold, proposals->get_attrs().nms_threshold);
    EXPECT_EQ(g_proposals->get_attrs().post_nms_count, proposals->get_attrs().post_nms_count);
    EXPECT_EQ(g_proposals->get_attrs().pre_nms_count, proposals->get_attrs().pre_nms_count);
}
