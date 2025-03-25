// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unique.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, unique_default_attributes) {
    NodeBuilder::opset().insert<ov::op::v10::Unique>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 10, 10});
    const auto grid = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 5, 5, 2});

    const auto op = make_shared<ov::op::v10::Unique>(data);
    NodeBuilder builder(op);
    auto g_unique = ov::as_type_ptr<ov::op::v10::Unique>(builder.create());

    constexpr auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(op->get_sorted(), g_unique->get_sorted());
    EXPECT_EQ(op->get_index_element_type(), g_unique->get_index_element_type());
    EXPECT_EQ(op->get_count_element_type(), g_unique->get_count_element_type());
}

TEST(attributes, unique_sorted_false) {
    NodeBuilder::opset().insert<ov::op::v10::Unique>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 10, 10});
    const auto grid = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 5, 5, 2});

    const auto op = make_shared<ov::op::v10::Unique>(data, false);
    NodeBuilder builder(op);
    auto g_unique = ov::as_type_ptr<ov::op::v10::Unique>(builder.create());

    constexpr auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(op->get_sorted(), g_unique->get_sorted());
    EXPECT_EQ(op->get_index_element_type(), g_unique->get_index_element_type());
    EXPECT_EQ(op->get_count_element_type(), g_unique->get_count_element_type());
}

TEST(attributes, unique_index_et_non_default) {
    NodeBuilder::opset().insert<ov::op::v10::Unique>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 10, 10});
    const auto grid = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 5, 5, 2});

    const auto op = make_shared<ov::op::v10::Unique>(data, true, element::i32, element::i32);
    NodeBuilder builder(op);
    auto g_unique = ov::as_type_ptr<ov::op::v10::Unique>(builder.create());

    constexpr auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(op->get_sorted(), g_unique->get_sorted());
    EXPECT_EQ(op->get_index_element_type(), g_unique->get_index_element_type());
    EXPECT_EQ(op->get_count_element_type(), g_unique->get_count_element_type());
}
