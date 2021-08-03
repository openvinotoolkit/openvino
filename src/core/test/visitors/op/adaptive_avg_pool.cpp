// Copyright (C) 2018-2021 Intel Corporation
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

TEST(attributes, adaptive_avg_pool_op)
{
    NodeBuilder::get_ops().register_factory<opset8::AdaptiveAvgPool>();
    const auto A = make_shared<op::Parameter>(element::f32, Shape{1, 3, 5, 4});
    const auto out_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {4, 3});

    const auto adaptive_pool = make_shared<opset8::AdaptiveAvgPool>(A, out_shape);
    NodeBuilder builder(adaptive_pool);

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
