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

TEST(type_prop, lstm_sequence)
{
    const auto X = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
    const auto W = make_shared<op::Parameter>(element::f32, Shape{1, 12, 4});
    const auto R = make_shared<op::Parameter>(element::f32, Shape{1, 12, 3});
    const auto initial_hidden_state = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto initial_cell_state = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{1, 12});
    const auto sequence_lengths = make_shared<op::Parameter>(element::i32, Shape{2});
    const auto hidden_size = 3;

    const auto lstm_sequence = make_shared<op::LSTMSequence>(X,
                                                             initial_hidden_state,
                                                             initial_cell_state,
                                                             sequence_lengths,
                                                             W,
                                                             R,
                                                             B,
                                                             hidden_size,
                                                             op::LSTMSequence::direction::FORWARD);
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
    EXPECT_EQ(lstm_sequence->get_output_shape(0), (Shape{1, 1, 2, 3}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(1), element::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(1), (Shape{1, 2, 3}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(2), element::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(2), (Shape{1, 2, 3}));
}
