// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/random_poisson.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, random_poisson_op) {
    NodeBuilder::opset().insert<ov::op::v17::RandomPoisson>();
    auto input = make_shared<ov::op::v0::Constant>(element::f32,
                                                   Shape{2, 3, 4},
                                                   vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    const auto random_poisson = make_shared<ov::op::v17::RandomPoisson>(input, 120, 100, op::PhiloxAlignment::PYTORCH);
    NodeBuilder builder(random_poisson, {input});
    auto g_random_poisson = ov::as_type_ptr<ov::op::v17::RandomPoisson>(builder.create());

    const auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_random_poisson->get_global_seed(), random_poisson->get_global_seed());
    EXPECT_EQ(g_random_poisson->get_op_seed(), random_poisson->get_op_seed());
    EXPECT_EQ(g_random_poisson->get_alignment(), random_poisson->get_alignment());
}