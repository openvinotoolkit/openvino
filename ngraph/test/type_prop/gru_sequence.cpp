//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
#include "ngraph/opsets/opset5.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, gru_sequence_forward)
{
    const size_t batch_size = 8;
    const size_t num_directions = 1;
    const size_t seq_length = 6;
    const size_t input_size = 4;
    const size_t hidden_size = 128;

    const auto X =
        make_shared<opset5::Parameter>(element::f32, Shape{batch_size, seq_length, input_size});
    const auto initial_hidden_state = make_shared<opset5::Parameter>(
        element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<op::Parameter>(element::i32, Shape{batch_size});
    const auto W = make_shared<opset5::Parameter>(
        element::f32, Shape{num_directions, 3 * hidden_size, input_size});
    const auto R = make_shared<opset5::Parameter>(
        element::f32, Shape{num_directions, 3 * hidden_size, hidden_size});
    const auto B =
        make_shared<opset5::Parameter>(element::f32, Shape{num_directions, 3 * hidden_size});

    const auto direction = op::RecurrentSequenceDirection::FORWARD;

    const auto sequence = make_shared<opset5::GRUSequence>(
        X, initial_hidden_state, sequence_lengths, W, R, B, hidden_size, direction);

    EXPECT_EQ(sequence->get_hidden_size(), hidden_size);
    EXPECT_EQ(sequence->get_direction(), op::RecurrentSequenceDirection::FORWARD);
    EXPECT_TRUE(sequence->get_activations_alpha().empty());
    EXPECT_TRUE(sequence->get_activations_beta().empty());
    EXPECT_EQ(sequence->get_activations()[0], "sigmoid");
    EXPECT_EQ(sequence->get_activations()[1], "tanh");
    EXPECT_EQ(sequence->get_clip(), 0.f);
    EXPECT_EQ(sequence->get_linear_before_reset(), false);
    EXPECT_EQ(sequence->get_output_element_type(0), element::f32);
    EXPECT_EQ(sequence->outputs().size(), 2);
    EXPECT_EQ(sequence->get_output_shape(0),
              (Shape{batch_size, num_directions, seq_length, hidden_size}));
    EXPECT_EQ(sequence->get_output_element_type(1), element::f32);
    EXPECT_EQ(sequence->get_output_shape(1), (Shape{batch_size, num_directions, hidden_size}));
}
