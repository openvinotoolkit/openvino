// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/random_uniform.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, random_uniform_op) {
    NodeBuilder::opset().insert<ov::op::v8::RandomUniform>();
    auto out_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{3}, vector<int64_t>{3, 2, 4});
    auto min_val = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, 0);
    auto max_val = make_shared<ov::op::v0::Constant>(element::f32, Shape{}, 1);

    const auto random_uniform = make_shared<ov::op::v8::RandomUniform>(out_shape,
                                                                       min_val,
                                                                       max_val,
                                                                       element::Type_t::f32,
                                                                       150,
                                                                       10,
                                                                       ov::op::PhiloxAlignment::TENSORFLOW);
    NodeBuilder builder(random_uniform, {out_shape, min_val, max_val});
    auto g_random_uniform = ov::as_type_ptr<ov::op::v8::RandomUniform>(builder.create());

    const auto expected_attr_count = 4;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_random_uniform->get_global_seed(), random_uniform->get_global_seed());
    EXPECT_EQ(g_random_uniform->get_op_seed(), random_uniform->get_op_seed());
    EXPECT_EQ(g_random_uniform->get_out_type(), random_uniform->get_out_type());
    EXPECT_EQ(g_random_uniform->get_alignment(), random_uniform->get_alignment());
}
