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
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, topk_op) {
    NodeBuilder::get_ops().register_factory<opset1::TopK>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4, 5});
    auto k = make_shared<op::Parameter>(element::i32, Shape{});

    auto axis = 0;
    auto mode = opset1::TopK::Mode::MAX;
    auto sort_type = opset1::TopK::SortType::SORT_VALUES;

    auto topk = make_shared<opset1::TopK>(data, k, axis, mode, sort_type);
    NodeBuilder builder(topk);
    auto g_topk = ov::as_type_ptr<opset1::TopK>(builder.create());

    EXPECT_EQ(g_topk->get_axis(), topk->get_axis());
    EXPECT_EQ(g_topk->get_mode(), topk->get_mode());
    EXPECT_EQ(g_topk->get_sort_type(), topk->get_sort_type());
}
