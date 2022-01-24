// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, idft_op) {
    NodeBuilder::get_ops().register_factory<op::v7::IDFT>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 10, 10, 2});
    auto axes = op::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {2});
    auto idft = make_shared<op::v7::IDFT>(data, axes);

    NodeBuilder builder(idft);
    const auto expected_attr_count = 0;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, idft_op_signal) {
    NodeBuilder::get_ops().register_factory<op::v7::IDFT>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 10, 10, 2});
    auto axes = op::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {2});
    auto signal = op::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {20});
    auto idft = make_shared<op::v7::IDFT>(data, axes, signal);

    NodeBuilder builder(idft);
    const auto expected_attr_count = 0;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}