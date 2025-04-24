// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lstm_cell.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, lstm_cell_v0_op) {
    NodeBuilder::opset().insert<ov::op::v0::LSTMCell>();
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
    auto H = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
    auto W = make_shared<ov::op::v0::Parameter>(element::f32, Shape{12, 3});
    auto R = make_shared<ov::op::v0::Parameter>(element::f32, Shape{12, 3});
    const auto initial_hidden_state = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
    const auto initial_cell_state = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});

    const auto hidden_size = 3;
    auto weights_format = ov::op::LSTMWeightsFormat::IFCO;
    const std::vector<std::string> activations = {"tanh", "sigmoid", "tanh"};
    auto activations_alpha = std::vector<float>{1.0, 1.5};
    auto activations_beta = std::vector<float>{2.0, 1.0};
    const float clip = 0.5f;
    auto input_forget = false;
    const auto lstm_cell = make_shared<ov::op::v0::LSTMCell>(X,
                                                             initial_hidden_state,
                                                             initial_cell_state,
                                                             W,
                                                             R,
                                                             hidden_size,
                                                             weights_format,
                                                             activations,
                                                             activations_alpha,
                                                             activations_beta,
                                                             clip,
                                                             input_forget);
    NodeBuilder builder(lstm_cell, {X, initial_hidden_state, initial_cell_state, W, R});
    auto g_lstm_cell = ov::as_type_ptr<ov::op::v0::LSTMCell>(builder.create());

    EXPECT_EQ(g_lstm_cell->get_hidden_size(), lstm_cell->get_hidden_size());
    EXPECT_EQ(g_lstm_cell->get_activations(), lstm_cell->get_activations());
    EXPECT_EQ(g_lstm_cell->get_activations_alpha(), lstm_cell->get_activations_alpha());
    EXPECT_EQ(g_lstm_cell->get_activations_beta(), lstm_cell->get_activations_beta());
    EXPECT_EQ(g_lstm_cell->get_clip(), lstm_cell->get_clip());
}

TEST(attributes, lstm_cell_v4_op) {
    NodeBuilder::opset().insert<ov::op::v4::LSTMCell>();
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
    auto H = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
    auto W = make_shared<ov::op::v0::Parameter>(element::f32, Shape{12, 3});
    auto R = make_shared<ov::op::v0::Parameter>(element::f32, Shape{12, 3});
    const auto initial_hidden_state = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
    const auto initial_cell_state = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});

    const auto hidden_size = 3;
    const std::vector<std::string> activations = {"tanh", "sigmoid", "tanh"};
    auto activations_alpha = std::vector<float>{1.0, 1.5};
    auto activations_beta = std::vector<float>{2.0, 1.0};
    const float clip = 0.5f;
    const auto lstm_cell = make_shared<ov::op::v4::LSTMCell>(X,
                                                             initial_hidden_state,
                                                             initial_cell_state,
                                                             W,
                                                             R,
                                                             hidden_size,
                                                             activations,
                                                             activations_alpha,
                                                             activations_beta,
                                                             clip);
    NodeBuilder builder(lstm_cell, {X, initial_hidden_state, initial_cell_state, W, R});
    auto g_lstm_cell = ov::as_type_ptr<ov::op::v4::LSTMCell>(builder.create());

    EXPECT_EQ(g_lstm_cell->get_hidden_size(), lstm_cell->get_hidden_size());
    EXPECT_EQ(g_lstm_cell->get_activations(), lstm_cell->get_activations());
    EXPECT_EQ(g_lstm_cell->get_activations_alpha(), lstm_cell->get_activations_alpha());
    EXPECT_EQ(g_lstm_cell->get_activations_beta(), lstm_cell->get_activations_beta());
    EXPECT_EQ(g_lstm_cell->get_clip(), lstm_cell->get_clip());
}

TEST(attributes, lstm_cell_v4_op2) {
    NodeBuilder::opset().insert<ov::op::v4::LSTMCell>();
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
    auto H = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
    auto W = make_shared<ov::op::v0::Parameter>(element::f32, Shape{12, 3});
    auto R = make_shared<ov::op::v0::Parameter>(element::f32, Shape{12, 3});
    const auto initial_hidden_state = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
    const auto initial_cell_state = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{12});

    const auto hidden_size = 3;
    const std::vector<std::string> activations = {"tanh", "sigmoid", "tanh"};
    auto activations_alpha = std::vector<float>{1.0, 1.5};
    auto activations_beta = std::vector<float>{2.0, 1.0};
    const float clip = 0.5f;
    const auto lstm_cell = make_shared<ov::op::v4::LSTMCell>(X,
                                                             initial_hidden_state,
                                                             initial_cell_state,
                                                             W,
                                                             R,
                                                             B,
                                                             hidden_size,
                                                             activations,
                                                             activations_alpha,
                                                             activations_beta,
                                                             clip);
    NodeBuilder builder(lstm_cell, {X, initial_hidden_state, initial_cell_state, W, R, B});
    auto g_lstm_cell = ov::as_type_ptr<ov::op::v4::LSTMCell>(builder.create());

    EXPECT_EQ(g_lstm_cell->get_hidden_size(), lstm_cell->get_hidden_size());
    EXPECT_EQ(g_lstm_cell->get_activations(), lstm_cell->get_activations());
    EXPECT_EQ(g_lstm_cell->get_activations_alpha(), lstm_cell->get_activations_alpha());
    EXPECT_EQ(g_lstm_cell->get_activations_beta(), lstm_cell->get_activations_beta());
    EXPECT_EQ(g_lstm_cell->get_clip(), lstm_cell->get_clip());
}
