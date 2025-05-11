// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/dft.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, dft_op) {
    NodeBuilder::opset().insert<op::v7::DFT>();
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{2, 10, 10, 2});
    auto axes = op::v0::Constant::create<int64_t>(element::i64, Shape{1}, {2});
    auto dft = make_shared<op::v7::DFT>(data, axes);

    NodeBuilder builder(dft, {data, axes});
    const auto expected_attr_count = 0;
    EXPECT_NO_THROW(auto g_dft = ov::as_type_ptr<op::v7::DFT>(builder.create()));

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, dft_op_signal) {
    NodeBuilder::opset().insert<op::v7::DFT>();
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{2, 10, 10, 2});
    auto signal = ov::op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {20});
    auto axes = op::v0::Constant::create<int64_t>(element::i64, Shape{1}, {2});
    auto dft = make_shared<op::v7::DFT>(data, axes, signal);

    NodeBuilder builder(dft, {data, axes, signal});
    EXPECT_NO_THROW(auto g_dft = ov::as_type_ptr<op::v7::DFT>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
