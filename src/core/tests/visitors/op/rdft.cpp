// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/rdft.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, rdft_op) {
    NodeBuilder::opset().insert<op::v9::RDFT>();
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{2, 10, 10});
    auto axes = op::v0::Constant::create<int64_t>(element::i64, Shape{1}, {2});
    auto rdft = make_shared<op::v9::RDFT>(data, axes);

    NodeBuilder builder(rdft, {data, axes});
    EXPECT_NO_THROW(auto g_rdft = ov::as_type_ptr<op::v9::RDFT>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, rdft_op_signal) {
    NodeBuilder::opset().insert<op::v9::RDFT>();
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{2, 10, 10});
    auto signal = op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {20});
    auto axes = op::v0::Constant::create<int64_t>(element::i64, Shape{1}, {2});
    auto rdft = make_shared<op::v9::RDFT>(data, axes, signal);

    NodeBuilder builder(rdft, {data, axes, signal});
    EXPECT_NO_THROW(auto g_rdft = ov::as_type_ptr<op::v9::RDFT>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
