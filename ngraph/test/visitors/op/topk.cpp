// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/topk.hpp"

#include "gtest/gtest.h"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, topk_op) {
    NodeBuilder::get_ops().register_factory<op::v1::TopK>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4, 5});
    auto k = make_shared<op::Parameter>(element::i32, Shape{});

    auto axis = 0;
    auto mode = op::v1::TopK::Mode::MAX;
    auto sort_type = op::v1::TopK::SortType::SORT_VALUES;

    auto topk = make_shared<op::v1::TopK>(data, k, axis, mode, sort_type);
    NodeBuilder builder(topk);
    auto g_topk = as_type_ptr<op::v1::TopK>(builder.create());

    EXPECT_EQ(g_topk->get_axis(), topk->get_axis());
    EXPECT_EQ(g_topk->get_mode(), topk->get_mode());
    EXPECT_EQ(g_topk->get_sort_type(), topk->get_sort_type());
}
