// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset2.hpp"
#include "ngraph/opsets/opset7.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, gelu_op_erf) {
    NodeBuilder::get_ops().register_factory<opset7::Gelu>();
    const auto data_input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto approximation_mode = op::GeluApproximationMode::ERF;
    const auto gelu = make_shared<opset7::Gelu>(data_input, approximation_mode);
    NodeBuilder builder(gelu, {data_input});
    auto g_gelu = ov::as_type_ptr<opset7::Gelu>(builder.create());

    EXPECT_EQ(g_gelu->get_approximation_mode(), gelu->get_approximation_mode());
}

TEST(attributes, gelu_op_tanh) {
    NodeBuilder::get_ops().register_factory<opset7::Gelu>();
    const auto data_input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto approximation_mode = op::GeluApproximationMode::TANH;
    const auto gelu = make_shared<opset7::Gelu>(data_input, approximation_mode);
    NodeBuilder builder(gelu, {data_input});
    auto g_gelu = ov::as_type_ptr<opset7::Gelu>(builder.create());

    EXPECT_EQ(g_gelu->get_approximation_mode(), gelu->get_approximation_mode());
}

TEST(attributes, gelu_op) {
    NodeBuilder::get_ops().register_factory<opset7::Gelu>();
    const auto data_input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto gelu = make_shared<opset7::Gelu>(data_input);
    NodeBuilder builder(gelu, {data_input});
    auto g_gelu = ov::as_type_ptr<opset7::Gelu>(builder.create());

    EXPECT_EQ(g_gelu->get_approximation_mode(), gelu->get_approximation_mode());
}

TEST(attributes, gelu_v0_op) {
    NodeBuilder::get_ops().register_factory<opset2::Gelu>();
    const auto data_input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto gelu = make_shared<opset2::Gelu>(data_input);
    NodeBuilder builder(gelu, {data_input});
    const auto expected_attr_count = 0;
    EXPECT_NO_THROW(auto g_gelu = ov::as_type_ptr<op::v7::DFT>(builder.create()));

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
