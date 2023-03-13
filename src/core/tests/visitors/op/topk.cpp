// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset11.hpp"
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
    NodeBuilder builder(topk, {data, k});
    auto g_topk = ov::as_type_ptr<opset1::TopK>(builder.create());

    EXPECT_EQ(g_topk->get_axis(), topk->get_axis());
    EXPECT_EQ(g_topk->get_mode(), topk->get_mode());
    EXPECT_EQ(g_topk->get_sort_type(), topk->get_sort_type());
    EXPECT_EQ(g_topk->get_index_element_type(), topk->get_index_element_type());
}

TEST(attributes, topk_v3_op) {
    NodeBuilder::get_ops().register_factory<opset3::TopK>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4, 5});
    auto k = make_shared<op::Parameter>(element::i32, Shape{});

    auto axis = 0;
    auto mode = opset3::TopK::Mode::MAX;
    auto sort_type = opset3::TopK::SortType::SORT_VALUES;

    auto topk = make_shared<opset3::TopK>(data, k, axis, mode, sort_type);
    NodeBuilder builder(topk, {data, k});
    auto g_topk = ov::as_type_ptr<opset3::TopK>(builder.create());

    EXPECT_EQ(g_topk->get_axis(), topk->get_axis());
    EXPECT_EQ(g_topk->get_mode(), topk->get_mode());
    EXPECT_EQ(g_topk->get_sort_type(), topk->get_sort_type());
    EXPECT_EQ(g_topk->get_index_element_type(), topk->get_index_element_type());
}

TEST(attributes, topk_v11_op) {
    NodeBuilder::get_ops().register_factory<ov::opset11::TopK>();
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 1, 3, 7});
    const auto k = make_shared<op::Parameter>(element::i32, Shape{});

    const auto axis = 0;
    const auto mode = ov::op::v11::TopK::Mode::MAX;
    const auto sort_type = ov::op::v11::TopK::SortType::SORT_VALUES;
    const auto idx_type = ov::element::i32;
    const auto stable = true;

    const auto topk = make_shared<ov::opset11::TopK>(data, k, axis, mode, sort_type, idx_type, stable);
    NodeBuilder builder(topk, {data, k});
    const auto g_topk = ov::as_type_ptr<ov::opset11::TopK>(builder.create());

    EXPECT_EQ(g_topk->get_axis(), topk->get_axis());
    EXPECT_EQ(g_topk->get_mode(), topk->get_mode());
    EXPECT_EQ(g_topk->get_sort_type(), topk->get_sort_type());
    EXPECT_EQ(g_topk->get_index_element_type(), topk->get_index_element_type());
    EXPECT_EQ(g_topk->get_stable(), topk->get_stable());
}
