//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, lstm_sequence_forward)
{
    const auto batch_size = 8;
    const auto num_directions = 1;
    const auto seq_length = 6;
    const auto input_size = 4;
    const auto hidden_size = 128;

    const auto X =
        make_shared<op::Parameter>(element::f32, Shape{batch_size, seq_length, input_size});
    const auto initial_hidden_state =
        make_shared<op::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto initial_cell_state =
        make_shared<op::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<op::Parameter>(element::i32, Shape{batch_size});
    const auto W = make_shared<op::Parameter>(element::f32,
                                              Shape{num_directions, 4 * hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32,
                                              Shape{num_directions, 4 * hidden_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{num_directions, 4 * hidden_size});

    const auto lstm_direction = op::LSTMSequence::direction::FORWARD;

    const auto lstm_sequence = make_shared<op::LSTMSequence>(X,
                                                             initial_hidden_state,
                                                             initial_cell_state,
                                                             sequence_lengths,
                                                             W,
                                                             R,
                                                             B,
                                                             hidden_size,
                                                             lstm_direction);
    EXPECT_EQ(lstm_sequence->get_hidden_size(), hidden_size);
    EXPECT_EQ(lstm_sequence->get_direction(), op::LSTMSequence::direction::FORWARD);
    EXPECT_EQ(lstm_sequence->get_weights_format(), op::LSTMWeightsFormat::IFCO);
    EXPECT_TRUE(lstm_sequence->get_activations_alpha().empty());
    EXPECT_TRUE(lstm_sequence->get_activations_beta().empty());
    EXPECT_EQ(lstm_sequence->get_activations()[0], "sigmoid");
    EXPECT_EQ(lstm_sequence->get_activations()[1], "tanh");
    EXPECT_EQ(lstm_sequence->get_activations()[2], "tanh");
    EXPECT_EQ(lstm_sequence->get_clip_threshold(), 0.f);
    EXPECT_FALSE(lstm_sequence->get_input_forget());
    EXPECT_EQ(lstm_sequence->get_output_element_type(0), element::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(0),
              (Shape{batch_size, num_directions, seq_length, hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(1), element::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(1), (Shape{batch_size, num_directions, hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(2), element::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(2), (Shape{batch_size, num_directions, hidden_size}));
}

TEST(type_prop, lstm_sequence_bidirectional)
{
    const auto batch_size = 24;
    const auto num_directions = 2;
    const auto seq_length = 12;
    const auto input_size = 8;
    const auto hidden_size = 256;

    const auto X =
        make_shared<op::Parameter>(element::f32, Shape{batch_size, seq_length, input_size});
    const auto initial_hidden_state =
        make_shared<op::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto initial_cell_state =
        make_shared<op::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<op::Parameter>(element::i32, Shape{batch_size});
    const auto W = make_shared<op::Parameter>(element::f32,
                                              Shape{num_directions, 4 * hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32,
                                              Shape{num_directions, 4 * hidden_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{num_directions, 4 * hidden_size});

    const auto weights_format = op::LSTMWeightsFormat::FICO;
    const auto lstm_direction = op::LSTMSequence::direction::BIDIRECTIONAL;
    const std::vector<float> activations_alpha = {2.7, 7.0, 32.367};
    const std::vector<float> activations_beta = {0.0, 5.49, 6.0};
    const std::vector<std::string> activations = {"tanh", "sigmoid", "sigmoid"};

    const auto lstm_sequence = make_shared<op::LSTMSequence>(X,
                                                             initial_hidden_state,
                                                             initial_cell_state,
                                                             sequence_lengths,
                                                             W,
                                                             R,
                                                             B,
                                                             hidden_size,
                                                             lstm_direction,
                                                             weights_format,
                                                             activations_alpha,
                                                             activations_beta,
                                                             activations);
    EXPECT_EQ(lstm_sequence->get_hidden_size(), hidden_size);
    EXPECT_EQ(lstm_sequence->get_direction(), op::LSTMSequence::direction::BIDIRECTIONAL);
    EXPECT_EQ(lstm_sequence->get_weights_format(), op::LSTMWeightsFormat::FICO);
    EXPECT_EQ(lstm_sequence->get_activations_alpha(), activations_alpha);
    EXPECT_EQ(lstm_sequence->get_activations_beta(), activations_beta);
    EXPECT_EQ(lstm_sequence->get_activations()[0], "tanh");
    EXPECT_EQ(lstm_sequence->get_activations()[1], "sigmoid");
    EXPECT_EQ(lstm_sequence->get_activations()[2], "sigmoid");
    EXPECT_EQ(lstm_sequence->get_clip_threshold(), 0.f);
    EXPECT_FALSE(lstm_sequence->get_input_forget());
    EXPECT_EQ(lstm_sequence->get_output_element_type(0), element::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(0),
              (Shape{batch_size, num_directions, seq_length, hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(1), element::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(1), (Shape{batch_size, num_directions, hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(2), element::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(2), (Shape{batch_size, num_directions, hidden_size}));
}