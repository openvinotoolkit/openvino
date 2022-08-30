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

TEST(attributes, rdft_op) {
    NodeBuilder::get_ops().register_factory<op::v9::RDFT>();
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{2, 10, 10});
    auto axes = op::v0::Constant::create<int64_t>(element::i64, Shape{1}, {2});
    auto rdft = make_shared<op::v9::RDFT>(data, axes);

    NodeBuilder builder(rdft);
    const auto expected_attr_count = 0;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, rdft_op_signal) {
    NodeBuilder::get_ops().register_factory<op::v9::RDFT>();
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{2, 10, 10});
    auto signal = op::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {20});
    auto axes = op::v0::Constant::create<int64_t>(element::i64, Shape{1}, {2});
    auto rdft = make_shared<op::v9::RDFT>(data, axes, signal);

    NodeBuilder builder(rdft);
    const auto expected_attr_count = 0;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}