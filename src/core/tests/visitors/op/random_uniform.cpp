// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset8.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, random_uniform_op) {
    NodeBuilder::get_ops().register_factory<opset8::RandomUniform>();
    auto out_shape = make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{3, 2, 4});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, 0);
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 1);

    const auto random_uniform =
        make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::f32, 150, 10);
    NodeBuilder builder(random_uniform);
    auto g_random_uniform = ov::as_type_ptr<opset8::RandomUniform>(builder.create());

    const auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_random_uniform->get_global_seed(), random_uniform->get_global_seed());
    EXPECT_EQ(g_random_uniform->get_op_seed(), random_uniform->get_op_seed());
    EXPECT_EQ(g_random_uniform->get_out_type(), random_uniform->get_out_type());
}
