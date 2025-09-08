// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/topk.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, topk_op) {
    NodeBuilder::opset().insert<ov::op::v1::TopK>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 3, 4, 5});
    auto k = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});

    auto axis = 0;
    auto mode = ov::op::v1::TopK::Mode::MAX;
    auto sort_type = ov::op::v1::TopK::SortType::SORT_VALUES;

    auto topk = make_shared<ov::op::v1::TopK>(data, k, axis, mode, sort_type);
    NodeBuilder builder(topk, {data, k});
    auto g_topk = ov::as_type_ptr<ov::op::v1::TopK>(builder.create());

    EXPECT_EQ(g_topk->get_axis(), topk->get_axis());
    EXPECT_EQ(g_topk->get_mode(), topk->get_mode());
    EXPECT_EQ(g_topk->get_sort_type(), topk->get_sort_type());
    EXPECT_EQ(g_topk->get_index_element_type(), topk->get_index_element_type());
}

TEST(attributes, topk_v3_op) {
    NodeBuilder::opset().insert<ov::op::v3::TopK>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 3, 4, 5});
    auto k = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});

    auto axis = 0;
    auto mode = ov::op::v3::TopK::Mode::MAX;
    auto sort_type = ov::op::v3::TopK::SortType::SORT_VALUES;

    auto topk = make_shared<ov::op::v3::TopK>(data, k, axis, mode, sort_type);
    NodeBuilder builder(topk, {data, k});
    auto g_topk = ov::as_type_ptr<ov::op::v3::TopK>(builder.create());

    EXPECT_EQ(g_topk->get_axis(), topk->get_axis());
    EXPECT_EQ(g_topk->get_mode(), topk->get_mode());
    EXPECT_EQ(g_topk->get_sort_type(), topk->get_sort_type());
    EXPECT_EQ(g_topk->get_index_element_type(), topk->get_index_element_type());
}

TEST(attributes, topk_v11_op) {
    NodeBuilder::opset().insert<ov::op::v11::TopK>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 1, 3, 7});
    const auto k = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});

    const auto axis = 0;
    const auto mode = ov::op::v11::TopK::Mode::MAX;
    const auto sort_type = ov::op::v11::TopK::SortType::SORT_VALUES;
    const auto idx_type = ov::element::i32;
    const auto stable = true;

    const auto topk = make_shared<ov::op::v11::TopK>(data, k, axis, mode, sort_type, idx_type, stable);
    NodeBuilder builder(topk, {data, k});
    const auto g_topk = ov::as_type_ptr<ov::op::v11::TopK>(builder.create());

    EXPECT_EQ(g_topk->get_axis(), topk->get_axis());
    EXPECT_EQ(g_topk->get_mode(), topk->get_mode());
    EXPECT_EQ(g_topk->get_sort_type(), topk->get_sort_type());
    EXPECT_EQ(g_topk->get_index_element_type(), topk->get_index_element_type());
    EXPECT_EQ(g_topk->get_stable(), topk->get_stable());
}
