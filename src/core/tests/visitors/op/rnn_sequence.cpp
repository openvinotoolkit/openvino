// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, rnn_sequence_op) {
    NodeBuilder::get_ops().register_factory<opset5::RNNSequence>();

    const size_t batch_size = 4;
    const size_t num_directions = 2;
    const size_t seq_length = 8;
    const size_t input_size = 16;
    const size_t hidden_size = 64;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, seq_length, input_size});
    const auto initial_hidden_state =
        make_shared<op::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<op::Parameter>(element::i32, Shape{batch_size});
    const auto W = make_shared<op::Parameter>(element::f32, Shape{num_directions, hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32, Shape{num_directions, hidden_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{num_directions, hidden_size});

    const auto rnn_direction = op::RecurrentSequenceDirection::BIDIRECTIONAL;
    const std::vector<float> activations_alpha = {1, 2, 3};
    const std::vector<float> activations_beta = {4, 5, 6};
    const std::vector<std::string> activations = {"tanh", "sigmoid", "tanh"};
    const float clip_threshold = 0.5f;

    const auto rnn_sequence = make_shared<opset5::RNNSequence>(X,
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
    NodeBuilder builder(rnn_sequence);
    auto g_rnn_sequence = ov::as_type_ptr<opset5::RNNSequence>(builder.create());

    EXPECT_EQ(g_rnn_sequence->get_hidden_size(), rnn_sequence->get_hidden_size());
    EXPECT_EQ(g_rnn_sequence->get_activations(), rnn_sequence->get_activations());
    EXPECT_EQ(g_rnn_sequence->get_activations_alpha(), rnn_sequence->get_activations_alpha());
    EXPECT_EQ(g_rnn_sequence->get_activations_beta(), rnn_sequence->get_activations_beta());
    EXPECT_EQ(g_rnn_sequence->get_clip(), rnn_sequence->get_clip());
    EXPECT_EQ(g_rnn_sequence->get_direction(), rnn_sequence->get_direction());
}
