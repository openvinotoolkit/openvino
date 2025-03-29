// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gelu.hpp"

#include <gtest/gtest.h>

#include "openvino/op/dft.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, gelu_op_erf) {
    NodeBuilder::opset().insert<ov::op::v7::Gelu>();
    const auto data_input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    const auto approximation_mode = op::GeluApproximationMode::ERF;
    const auto gelu = make_shared<ov::op::v7::Gelu>(data_input, approximation_mode);
    NodeBuilder builder(gelu, {data_input});
    auto g_gelu = ov::as_type_ptr<ov::op::v7::Gelu>(builder.create());

    EXPECT_EQ(g_gelu->get_approximation_mode(), gelu->get_approximation_mode());
}

TEST(attributes, gelu_op_tanh) {
    NodeBuilder::opset().insert<ov::op::v7::Gelu>();
    const auto data_input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    const auto approximation_mode = op::GeluApproximationMode::TANH;
    const auto gelu = make_shared<ov::op::v7::Gelu>(data_input, approximation_mode);
    NodeBuilder builder(gelu, {data_input});
    auto g_gelu = ov::as_type_ptr<ov::op::v7::Gelu>(builder.create());

    EXPECT_EQ(g_gelu->get_approximation_mode(), gelu->get_approximation_mode());
}

TEST(attributes, gelu_op) {
    NodeBuilder::opset().insert<ov::op::v7::Gelu>();
    const auto data_input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    const auto gelu = make_shared<ov::op::v7::Gelu>(data_input);
    NodeBuilder builder(gelu, {data_input});
    auto g_gelu = ov::as_type_ptr<ov::op::v7::Gelu>(builder.create());

    EXPECT_EQ(g_gelu->get_approximation_mode(), gelu->get_approximation_mode());
}

TEST(attributes, gelu_v0_op) {
    NodeBuilder::opset().insert<ov::op::v0::Gelu>();
    const auto data_input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    const auto gelu = make_shared<ov::op::v0::Gelu>(data_input);
    NodeBuilder builder(gelu, {data_input});
    const auto expected_attr_count = 0;
    EXPECT_NO_THROW(auto g_gelu = ov::as_type_ptr<op::v7::DFT>(builder.create()));

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
