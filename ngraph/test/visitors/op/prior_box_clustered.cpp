// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "util/visitor.hpp"
#include "ngraph/opsets/opset1.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, prior_box_clustered_op)
{
    NodeBuilder::get_ops().register_factory<opset1::PriorBoxClustered>();
    const auto layer_shape = make_shared<op::Parameter>(element::i64, Shape{32, 32});
    const auto image_shape = make_shared<op::Parameter>(element::i64, Shape{300, 300});

    op::PriorBoxClusteredAttrs attrs;
    attrs.heights = {2.0f, 3.0f};
    attrs.widths = {2.0f, 3.0f};
    attrs.clip = true;
    attrs.step_widths = 0.0f;
    attrs.step_heights = 0.0f;
    attrs.step = 0.0f;
    attrs.offset = 0.0f;
    attrs.variances = {0.1f};

    auto pbc = make_shared<opset1::PriorBoxClustered>(layer_shape, image_shape, attrs);
    NodeBuilder builder(pbc);
    auto g_pbc = as_type_ptr<opset1::PriorBoxClustered>(builder.create());
    const auto pbc_attrs = pbc->get_attrs();
    const auto g_pbc_attrs = g_pbc->get_attrs();
    const auto expected_attr_count = 8;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_pbc_attrs.heights, pbc_attrs.heights);
    EXPECT_EQ(g_pbc_attrs.widths, pbc_attrs.widths);
    EXPECT_EQ(g_pbc_attrs.clip, pbc_attrs.clip);
    EXPECT_EQ(g_pbc_attrs.step_widths, pbc_attrs.step_widths);
    EXPECT_EQ(g_pbc_attrs.step_heights, pbc_attrs.step_heights);
    EXPECT_EQ(g_pbc_attrs.offset, pbc_attrs.offset);
    EXPECT_EQ(g_pbc_attrs.variances, pbc_attrs.variances);
}
