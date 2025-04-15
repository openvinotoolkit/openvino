// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/einsum.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, einsum_v7_op) {
    NodeBuilder::opset().insert<ov::op::v7::Einsum>();
    auto input1 = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 3});
    auto input2 = make_shared<ov::op::v0::Parameter>(element::i32, Shape{3, 4});
    std::string equation = "ab,bc->ac";
    auto einsum = make_shared<ov::op::v7::Einsum>(OutputVector{input1, input2}, equation);
    NodeBuilder builder(einsum, {input1, input2});
    auto g_einsum = ov::as_type_ptr<ov::op::v7::Einsum>(builder.create());
    EXPECT_EQ(g_einsum->get_equation(), einsum->get_equation());
}
