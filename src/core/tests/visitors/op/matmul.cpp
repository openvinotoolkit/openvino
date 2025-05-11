// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/matmul.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, matmul_op) {
    NodeBuilder::opset().insert<ov::op::v0::MatMul>();
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{0, 2});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 0});

    bool transpose_a = true;
    bool transpose_b = true;

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, transpose_a, transpose_b);
    NodeBuilder builder(matmul, {A, B});
    auto g_matmul = ov::as_type_ptr<ov::op::v0::MatMul>(builder.create());

    EXPECT_EQ(g_matmul->get_transpose_a(), matmul->get_transpose_a());
    EXPECT_EQ(g_matmul->get_transpose_b(), matmul->get_transpose_b());
}

TEST(attributes, matmul_op2) {
    NodeBuilder::opset().insert<ov::op::v0::MatMul>();
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{10, 2});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 1});

    bool transpose_a = false;
    bool transpose_b = false;

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, transpose_a, transpose_b);
    NodeBuilder builder(matmul, {A, B});
    auto g_matmul = ov::as_type_ptr<ov::op::v0::MatMul>(builder.create());

    EXPECT_EQ(g_matmul->get_transpose_a(), matmul->get_transpose_a());
    EXPECT_EQ(g_matmul->get_transpose_b(), matmul->get_transpose_b());
}

TEST(attributes, matmul_op3) {
    NodeBuilder::opset().insert<ov::op::v0::MatMul>();
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 10});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 1});

    bool transpose_a = true;
    bool transpose_b = false;

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, transpose_a, transpose_b);
    NodeBuilder builder(matmul, {A, B});
    auto g_matmul = ov::as_type_ptr<ov::op::v0::MatMul>(builder.create());

    EXPECT_EQ(g_matmul->get_transpose_a(), matmul->get_transpose_a());
    EXPECT_EQ(g_matmul->get_transpose_b(), matmul->get_transpose_b());
}

TEST(attributes, matmul_op4) {
    NodeBuilder::opset().insert<ov::op::v0::MatMul>();
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 2});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 2, 2, 1});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B);
    NodeBuilder builder(matmul, {A, B});
    auto g_matmul = ov::as_type_ptr<ov::op::v0::MatMul>(builder.create());

    EXPECT_EQ(g_matmul->get_transpose_a(), matmul->get_transpose_a());
    EXPECT_EQ(g_matmul->get_transpose_b(), matmul->get_transpose_b());
}

TEST(attributes, matmul_op5) {
    NodeBuilder::opset().insert<ov::op::v0::MatMul>();
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 10});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B);
    NodeBuilder builder(matmul, {A, B});
    auto g_matmul = ov::as_type_ptr<ov::op::v0::MatMul>(builder.create());

    EXPECT_EQ(g_matmul->get_transpose_a(), matmul->get_transpose_a());
    EXPECT_EQ(g_matmul->get_transpose_b(), matmul->get_transpose_b());
}

TEST(attributes, matmul_op6) {
    NodeBuilder::opset().insert<ov::op::v0::MatMul>();
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2048});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2048, 1000});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B);
    NodeBuilder builder(matmul, {A, B});
    auto g_matmul = ov::as_type_ptr<ov::op::v0::MatMul>(builder.create());

    EXPECT_EQ(g_matmul->get_transpose_a(), matmul->get_transpose_a());
    EXPECT_EQ(g_matmul->get_transpose_b(), matmul->get_transpose_b());
}
