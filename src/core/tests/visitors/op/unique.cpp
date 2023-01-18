// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/opsets/opset10.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ov;
using ngraph::test::NodeBuilder;

TEST(attributes, unique_default_attributes) {
    NodeBuilder::get_ops().register_factory<opset10::Unique>();
    const auto data = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 10, 10});
    const auto grid = make_shared<opset10::Parameter>(element::f32, Shape{1, 5, 5, 2});

    const auto op = make_shared<opset10::Unique>(data);
    NodeBuilder builder(op);
    auto g_unique = ov::as_type_ptr<opset10::Unique>(builder.create());

    constexpr auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(op->get_sorted(), g_unique->get_sorted());
    EXPECT_EQ(op->get_index_element_type(), g_unique->get_index_element_type());
    EXPECT_EQ(op->get_count_element_type(), g_unique->get_count_element_type());
}

TEST(attributes, unique_sorted_false) {
    NodeBuilder::get_ops().register_factory<opset10::Unique>();
    const auto data = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 10, 10});
    const auto grid = make_shared<opset10::Parameter>(element::f32, Shape{1, 5, 5, 2});

    const auto op = make_shared<opset10::Unique>(data, false);
    NodeBuilder builder(op);
    auto g_unique = ov::as_type_ptr<opset10::Unique>(builder.create());

    constexpr auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(op->get_sorted(), g_unique->get_sorted());
    EXPECT_EQ(op->get_index_element_type(), g_unique->get_index_element_type());
    EXPECT_EQ(op->get_count_element_type(), g_unique->get_count_element_type());
}

TEST(attributes, unique_index_et_non_default) {
    NodeBuilder::get_ops().register_factory<opset10::Unique>();
    const auto data = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 10, 10});
    const auto grid = make_shared<opset10::Parameter>(element::f32, Shape{1, 5, 5, 2});

    const auto op = make_shared<opset10::Unique>(data, true, element::i32, element::i32);
    NodeBuilder builder(op);
    auto g_unique = ov::as_type_ptr<opset10::Unique>(builder.create());

    constexpr auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(op->get_sorted(), g_unique->get_sorted());
    EXPECT_EQ(op->get_index_element_type(), g_unique->get_index_element_type());
    EXPECT_EQ(op->get_count_element_type(), g_unique->get_count_element_type());
}
