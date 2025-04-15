// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/rnn_sequence.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, rnn_sequence_op) {
    NodeBuilder::opset().insert<ov::op::v5::RNNSequence>();

    const size_t batch_size = 4;
    const size_t num_directions = 2;
    const size_t seq_length = 8;
    const size_t input_size = 16;
    const size_t hidden_size = 64;

    const auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{batch_size, seq_length, input_size});
    const auto initial_hidden_state =
        make_shared<ov::op::v0::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<ov::op::v0::Parameter>(element::i32, Shape{batch_size});
    const auto W = make_shared<ov::op::v0::Parameter>(element::f32, Shape{num_directions, hidden_size, input_size});
    const auto R = make_shared<ov::op::v0::Parameter>(element::f32, Shape{num_directions, hidden_size, hidden_size});
    const auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{num_directions, hidden_size});

    const auto rnn_direction = op::RecurrentSequenceDirection::BIDIRECTIONAL;
    const std::vector<float> activations_alpha = {1, 2, 3};
    const std::vector<float> activations_beta = {4, 5, 6};
    const std::vector<std::string> activations = {"tanh", "sigmoid", "tanh"};
    const float clip_threshold = 0.5f;

    const auto rnn_sequence = make_shared<ov::op::v5::RNNSequence>(X,
                                                                   initial_hidden_state,
                                                                   sequence_lengths,
                                                                   W,
                                                                   R,
                                                                   B,
                                                                   hidden_size,
                                                                   rnn_direction,
                                                                   activations,
                                                                   activations_alpha,
                                                                   activations_beta,
                                                                   clip_threshold);
    NodeBuilder builder(rnn_sequence, {X, initial_hidden_state, sequence_lengths, W, R, B});
    auto g_rnn_sequence = ov::as_type_ptr<ov::op::v5::RNNSequence>(builder.create());

    EXPECT_EQ(g_rnn_sequence->get_hidden_size(), rnn_sequence->get_hidden_size());
    EXPECT_EQ(g_rnn_sequence->get_activations(), rnn_sequence->get_activations());
    EXPECT_EQ(g_rnn_sequence->get_activations_alpha(), rnn_sequence->get_activations_alpha());
    EXPECT_EQ(g_rnn_sequence->get_activations_beta(), rnn_sequence->get_activations_beta());
    EXPECT_EQ(g_rnn_sequence->get_clip(), rnn_sequence->get_clip());
    EXPECT_EQ(g_rnn_sequence->get_direction(), rnn_sequence->get_direction());
}
