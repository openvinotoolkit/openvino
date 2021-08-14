// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/op/matmul.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, matmul_op) {
    NodeBuilder::get_ops().register_factory<op::v0::MatMul>();
    auto A = make_shared<op::Parameter>(element::f32, Shape{0, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 0});

    bool transpose_a = true;
    bool transpose_b = true;

    auto matmul = make_shared<op::v0::MatMul>(A, B, transpose_a, transpose_b);
    NodeBuilder builder(matmul);
    auto g_matmul = as_type_ptr<op::v0::MatMul>(builder.create());

    EXPECT_EQ(g_matmul->get_transpose_a(), matmul->get_transpose_a());
    EXPECT_EQ(g_matmul->get_transpose_b(), matmul->get_transpose_b());
}
