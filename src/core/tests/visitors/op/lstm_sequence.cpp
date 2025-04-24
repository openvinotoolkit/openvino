// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lstm_sequence.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, lstm_sequence_op) {
    NodeBuilder::opset().insert<ov::op::v5::LSTMSequence>();

    const size_t batch_size = 4;
    const size_t num_directions = 2;
    const size_t seq_length = 8;
    const size_t input_size = 16;
    const size_t hidden_size = 64;

    const auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{batch_size, seq_length, input_size});
    const auto initial_hidden_state =
        make_shared<ov::op::v0::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto initial_cell_state =
        make_shared<ov::op::v0::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<ov::op::v0::Parameter>(element::i32, Shape{batch_size});
    const auto W = make_shared<ov::op::v0::Parameter>(element::f32, Shape{num_directions, 4 * hidden_size, input_size});
    const auto R =
        make_shared<ov::op::v0::Parameter>(element::f32, Shape{num_directions, 4 * hidden_size, hidden_size});
    const auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{num_directions, 4 * hidden_size});

    const auto lstm_direction = op::RecurrentSequenceDirection::BIDIRECTIONAL;
    const std::vector<float> activations_alpha = {1, 2, 3};
    const std::vector<float> activations_beta = {4, 5, 6};
    const std::vector<std::string> activations = {"tanh", "sigmoid", "tanh"};
    const float clip_threshold = 0.5f;

    const auto lstm_sequence = make_shared<ov::op::v5::LSTMSequence>(X,
                                                                     initial_hidden_state,
                                                                     initial_cell_state,
                                                                     sequence_lengths,
                                                                     W,
                                                                     R,
                                                                     B,
                                                                     hidden_size,
                                                                     lstm_direction,
                                                                     activations_alpha,
                                                                     activations_beta,
                                                                     activations,
                                                                     clip_threshold);
    NodeBuilder builder(lstm_sequence, {X, initial_hidden_state, initial_cell_state, sequence_lengths, W, R, B});
    auto g_lstm_sequence = ov::as_type_ptr<ov::op::v5::LSTMSequence>(builder.create());

    EXPECT_EQ(g_lstm_sequence->get_hidden_size(), lstm_sequence->get_hidden_size());
    EXPECT_EQ(g_lstm_sequence->get_activations(), lstm_sequence->get_activations());
    EXPECT_EQ(g_lstm_sequence->get_activations_alpha(), lstm_sequence->get_activations_alpha());
    EXPECT_EQ(g_lstm_sequence->get_activations_beta(), lstm_sequence->get_activations_beta());
    EXPECT_EQ(g_lstm_sequence->get_clip(), lstm_sequence->get_clip());
    EXPECT_EQ(g_lstm_sequence->get_direction(), lstm_sequence->get_direction());
}
