// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/rnn_cell.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, rnn_cell_op_custom_attributes) {
    NodeBuilder::opset().insert<ov::op::v0::RNNCell>();
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
    auto H = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
    auto W = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 3});
    auto R = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 3});

    const size_t hidden_size = 3;
    auto activations = std::vector<std::string>{"sigmoid", "tanh"};
    auto activations_alpha = std::vector<float>{1.0, 1.5};
    auto activations_beta = std::vector<float>{2.0, 1.0};
    float clip = 1.0;

    auto rnn_cell = make_shared<ov::op::v0::RNNCell>(X,
                                                     H,
                                                     W,
                                                     R,
                                                     hidden_size,
                                                     activations,
                                                     activations_alpha,
                                                     activations_beta,
                                                     clip);

    NodeBuilder builder(rnn_cell, {X, H, W, R});
    auto g_rnn_cell = ov::as_type_ptr<ov::op::v0::RNNCell>(builder.create());

    EXPECT_EQ(g_rnn_cell->get_hidden_size(), rnn_cell->get_hidden_size());
    EXPECT_EQ(g_rnn_cell->get_clip(), rnn_cell->get_clip());
    EXPECT_EQ(g_rnn_cell->get_activations(), rnn_cell->get_activations());
    EXPECT_EQ(g_rnn_cell->get_activations_alpha(), rnn_cell->get_activations_alpha());
    EXPECT_EQ(g_rnn_cell->get_activations_beta(), rnn_cell->get_activations_beta());
}

TEST(attributes, rnn_cell_op_default_attributes) {
    NodeBuilder::opset().insert<ov::op::v0::RNNCell>();
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
    auto H = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
    auto W = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 3});
    auto R = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 3});

    const size_t hidden_size = 3;

    auto rnn_cell = make_shared<ov::op::v0::RNNCell>(X, H, W, R, hidden_size);

    NodeBuilder builder(rnn_cell, {X, H, W, R});
    auto g_rnn_cell = ov::as_type_ptr<ov::op::v0::RNNCell>(builder.create());

    EXPECT_EQ(g_rnn_cell->get_hidden_size(), rnn_cell->get_hidden_size());
    EXPECT_EQ(g_rnn_cell->get_clip(), rnn_cell->get_clip());
    EXPECT_EQ(g_rnn_cell->get_activations(), rnn_cell->get_activations());
    EXPECT_EQ(g_rnn_cell->get_activations_alpha(), rnn_cell->get_activations_alpha());
    EXPECT_EQ(g_rnn_cell->get_activations_beta(), rnn_cell->get_activations_beta());
}

TEST(attributes, rnn_cell_op_default_attributes2) {
    NodeBuilder::opset().insert<ov::op::v0::RNNCell>();
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
    auto H = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
    auto W = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 3});
    auto R = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 3});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});

    const size_t hidden_size = 3;

    auto rnn_cell = make_shared<ov::op::v0::RNNCell>(X, H, W, R, B, hidden_size);

    NodeBuilder builder(rnn_cell, {X, H, W, R, B});
    auto g_rnn_cell = ov::as_type_ptr<ov::op::v0::RNNCell>(builder.create());

    EXPECT_EQ(g_rnn_cell->get_hidden_size(), rnn_cell->get_hidden_size());
    EXPECT_EQ(g_rnn_cell->get_clip(), rnn_cell->get_clip());
    EXPECT_EQ(g_rnn_cell->get_activations(), rnn_cell->get_activations());
    EXPECT_EQ(g_rnn_cell->get_activations_alpha(), rnn_cell->get_activations_alpha());
    EXPECT_EQ(g_rnn_cell->get_activations_beta(), rnn_cell->get_activations_beta());
}
