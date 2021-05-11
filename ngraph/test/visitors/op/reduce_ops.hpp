// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

template<class T>
class ReduceOpsAttrTest : public ::testing::Test
{
};

TYPED_TEST_CASE_P(ReduceOpsAttrTest);

TYPED_TEST_P(ReduceOpsAttrTest, reduce_ops)
{
    Shape in_shape{3, 4, 5};
    element::Type in_et = element::f32;

    Shape axes_shape{2};
    element::Type axes_et = element::i64;

    bool keep_dims = true;

    NodeBuilder::get_ops().register_factory<TypeParam>();
    auto data = make_shared<op::Parameter>(in_et, in_shape);
    auto reduction_axes = make_shared<op::Parameter>(axes_et, axes_shape);
    auto reduce_op = make_shared<TypeParam>(data, reduction_axes, keep_dims);

    NodeBuilder builder(reduce_op);
    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    auto g_reduce_op = as_type_ptr<TypeParam>(builder.create());
    EXPECT_EQ(g_reduce_op->get_keep_dims(), reduce_op->get_keep_dims());
}

REGISTER_TYPED_TEST_CASE_P(
    ReduceOpsAttrTest,
    reduce_ops);
