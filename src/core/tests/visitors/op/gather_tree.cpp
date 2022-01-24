// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "util/visitor.hpp"

using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, gather_tree_op) {
    NodeBuilder::get_ops().register_factory<opset1::GatherTree>();

    auto step_ids = std::make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto parent_idx = std::make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto max_seq_len = std::make_shared<op::Parameter>(element::f32, Shape{2});
    auto end_token = std::make_shared<op::Parameter>(element::f32, Shape{});

    auto gather_tree = std::make_shared<opset1::GatherTree>(step_ids, parent_idx, max_seq_len, end_token);
    NodeBuilder builder(gather_tree);

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}