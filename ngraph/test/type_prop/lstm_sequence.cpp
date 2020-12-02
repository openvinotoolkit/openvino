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
#include "ngraph/opsets/opset5.hpp"
#include "util/type_prop.hpp"

// suppress FusedOp deprecation warnings
NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

//
// RNN sequence parameters
//
struct recurrent_sequence_parameters
{
    Dimension batch_size = 24;
    Dimension num_directions = 2;
    Dimension seq_length = 12;
    Dimension input_size = 8;
    Dimension hidden_size = 256;
    ngraph::element::Type et = element::Type_t::f32;
};

//
// Create and initialize default input test tensors.
//
shared_ptr<opset5::LSTMSequence>
    lstm_seq_tensor_initialization(const recurrent_sequence_parameters& param)
{
    auto batch_size = param.batch_size;
    auto seq_length = param.seq_length;
    auto input_size = param.input_size;
    auto num_directions = param.num_directions;
    auto hidden_size = param.hidden_size;
    auto et = param.et;

    const auto X =
        make_shared<opset5::Parameter>(et, PartialShape{batch_size, seq_length, input_size});
    const auto initial_hidden_state =
        make_shared<opset5::Parameter>(et, PartialShape{batch_size, num_directions, hidden_size});
    const auto initial_cell_state =
        make_shared<opset5::Parameter>(et, PartialShape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<opset5::Parameter>(et, PartialShape{batch_size});
    const auto W = make_shared<opset5::Parameter>(
        et, PartialShape{num_directions, hidden_size * 4, input_size});
    const auto R = make_shared<opset5::Parameter>(
        et, PartialShape{num_directions, hidden_size * 4, hidden_size});
    const auto B =
        make_shared<opset5::Parameter>(et, PartialShape{num_directions, hidden_size * 4});

    const auto lstm_sequence = make_shared<opset5::LSTMSequence>();

    lstm_sequence->set_argument(0, X);
    lstm_sequence->set_argument(1, initial_hidden_state);
    lstm_sequence->set_argument(2, initial_cell_state);
    lstm_sequence->set_argument(3, sequence_lengths);
    lstm_sequence->set_argument(4, W);
    lstm_sequence->set_argument(5, R);
    lstm_sequence->set_argument(6, B);

    return lstm_sequence;
}

TEST(type_prop, lstm_sequence_forward)
{
    const size_t batch_size = 8;
    const size_t num_directions = 1;
    const size_t seq_length = 6;
    const size_t input_size = 4;
    const size_t hidden_size = 128;

    const auto X = make_shared<opset5::Parameter>(element::Type_t::f32,
                                                  Shape{batch_size, seq_length, input_size});
    const auto initial_hidden_state = make_shared<opset5::Parameter>(
        element::Type_t::f32, Shape{batch_size, num_directions, hidden_size});
    const auto initial_cell_state = make_shared<opset5::Parameter>(
        element::Type_t::f32, Shape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths =
        make_shared<opset5::Parameter>(element::Type_t::i32, Shape{batch_size});
    const auto W = make_shared<opset5::Parameter>(
        element::Type_t::f32, Shape{num_directions, 4 * hidden_size, input_size});
    const auto R = make_shared<opset5::Parameter>(
        element::Type_t::f32, Shape{num_directions, 4 * hidden_size, hidden_size});
    const auto B = make_shared<opset5::Parameter>(element::Type_t::f32,
                                                  Shape{num_directions, 4 * hidden_size});

    const auto lstm_direction = op::RecurrentSequenceDirection::FORWARD;

    const auto lstm_sequence = make_shared<opset5::LSTMSequence>(X,
                                                                 initial_hidden_state,
                                                                 initial_cell_state,
                                                                 sequence_lengths,
                                                                 W,
                                                                 R,
                                                                 B,
                                                                 hidden_size,
                                                                 lstm_direction);

    EXPECT_EQ(lstm_sequence->get_hidden_size(), hidden_size);
    EXPECT_EQ(lstm_sequence->get_direction(), op::RecurrentSequenceDirection::FORWARD);
    EXPECT_TRUE(lstm_sequence->get_activations_alpha().empty());
    EXPECT_TRUE(lstm_sequence->get_activations_beta().empty());
    EXPECT_EQ(lstm_sequence->get_activations()[0], "sigmoid");
    EXPECT_EQ(lstm_sequence->get_activations()[1], "tanh");
    EXPECT_EQ(lstm_sequence->get_activations()[2], "tanh");
    EXPECT_EQ(lstm_sequence->get_clip(), 0.f);
    EXPECT_EQ(lstm_sequence->get_output_element_type(0), element::Type_t::f32);
    EXPECT_EQ(lstm_sequence->outputs().size(), 3);
    EXPECT_EQ(lstm_sequence->get_output_shape(0),
              (Shape{batch_size, num_directions, seq_length, hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(1), element::Type_t::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(1), (Shape{batch_size, num_directions, hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(2), element::Type_t::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(2), (Shape{batch_size, num_directions, hidden_size}));
}

TEST(type_prop, lstm_sequence_bidirectional)
{
    const size_t batch_size = 24;
    const size_t num_directions = 2;
    const size_t seq_length = 12;
    const size_t input_size = 8;
    const size_t hidden_size = 256;

    const auto X = make_shared<opset5::Parameter>(element::Type_t::f32,
                                                  Shape{batch_size, seq_length, input_size});
    const auto initial_hidden_state = make_shared<opset5::Parameter>(
        element::Type_t::f32, Shape{batch_size, num_directions, hidden_size});
    const auto initial_cell_state = make_shared<opset5::Parameter>(
        element::Type_t::f32, Shape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths =
        make_shared<opset5::Parameter>(element::Type_t::i32, Shape{batch_size});
    const auto W = make_shared<opset5::Parameter>(
        element::Type_t::f32, Shape{num_directions, 4 * hidden_size, input_size});
    const auto R = make_shared<opset5::Parameter>(
        element::Type_t::f32, Shape{num_directions, 4 * hidden_size, hidden_size});
    const auto B = make_shared<opset5::Parameter>(element::Type_t::f32,
                                                  Shape{num_directions, 4 * hidden_size});

    const auto lstm_direction = opset5::LSTMSequence::direction::BIDIRECTIONAL;
    const std::vector<float> activations_alpha = {2.7, 7.0, 32.367};
    const std::vector<float> activations_beta = {0.0, 5.49, 6.0};
    const std::vector<std::string> activations = {"tanh", "sigmoid", "sigmoid"};

    const auto lstm_sequence = make_shared<opset5::LSTMSequence>(X,
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
                                                                 activations);
    EXPECT_EQ(lstm_sequence->get_hidden_size(), hidden_size);
    EXPECT_EQ(lstm_sequence->get_direction(), opset5::LSTMSequence::direction::BIDIRECTIONAL);
    EXPECT_EQ(lstm_sequence->get_activations_alpha(), activations_alpha);
    EXPECT_EQ(lstm_sequence->get_activations_beta(), activations_beta);
    EXPECT_EQ(lstm_sequence->get_activations()[0], "tanh");
    EXPECT_EQ(lstm_sequence->get_activations()[1], "sigmoid");
    EXPECT_EQ(lstm_sequence->get_activations()[2], "sigmoid");
    EXPECT_EQ(lstm_sequence->get_clip(), 0.f);
    EXPECT_EQ(lstm_sequence->get_output_element_type(0), element::Type_t::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(0),
              (Shape{batch_size, num_directions, seq_length, hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(1), element::Type_t::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(1), (Shape{batch_size, num_directions, hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(2), element::Type_t::f32);
    EXPECT_EQ(lstm_sequence->get_output_shape(2), (Shape{batch_size, num_directions, hidden_size}));
}

TEST(type_prop, lstm_sequence_dynamic_batch_size)
{
    recurrent_sequence_parameters param;

    param.batch_size = Dimension::dynamic();
    param.num_directions = 2;
    param.seq_length = 12;
    param.input_size = 8;
    param.hidden_size = 256;
    param.et = element::Type_t::f32;

    auto lstm_sequence = lstm_seq_tensor_initialization(param);
    lstm_sequence->validate_and_infer_types();

    EXPECT_EQ(lstm_sequence->get_output_partial_shape(0),
              (PartialShape{
                  param.batch_size, param.num_directions, param.seq_length, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(1),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(2),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(0), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(1), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(2), param.et);
}

TEST(type_prop, lstm_sequence_dynamic_num_directions)
{
    recurrent_sequence_parameters param;

    param.batch_size = 24;
    param.num_directions = Dimension::dynamic();
    param.seq_length = 12;
    param.input_size = 8;
    param.hidden_size = 256;
    param.et = element::Type_t::f32;

    auto lstm_sequence = lstm_seq_tensor_initialization(param);
    lstm_sequence->validate_and_infer_types();

    EXPECT_EQ(lstm_sequence->get_output_partial_shape(0),
              (PartialShape{
                  param.batch_size, param.num_directions, param.seq_length, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(1),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(2),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(0), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(1), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(2), param.et);
}

TEST(type_prop, lstm_sequence_dynamic_seq_length)
{
    recurrent_sequence_parameters param;

    param.batch_size = 24;
    param.num_directions = 2;
    param.seq_length = Dimension::dynamic();
    param.input_size = 8;
    param.hidden_size = 256;
    param.et = element::Type_t::f32;

    auto lstm_sequence = lstm_seq_tensor_initialization(param);
    lstm_sequence->validate_and_infer_types();

    EXPECT_EQ(lstm_sequence->get_output_partial_shape(0),
              (PartialShape{
                  param.batch_size, param.num_directions, param.seq_length, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(1),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(2),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(0), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(1), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(2), param.et);
}

TEST(type_prop, lstm_sequence_dynamic_hidden_size)
{
    recurrent_sequence_parameters param;

    param.batch_size = 24;
    param.num_directions = 2;
    param.seq_length = 12;
    param.input_size = 8;
    param.hidden_size = Dimension::dynamic();
    param.et = element::Type_t::f32;

    auto lstm_sequence = lstm_seq_tensor_initialization(param);
    lstm_sequence->validate_and_infer_types();

    EXPECT_EQ(lstm_sequence->get_output_partial_shape(0),
              (PartialShape{
                  param.batch_size, param.num_directions, param.seq_length, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(1),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(2),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(0), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(1), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(2), param.et);
}

TEST(type_prop, lstm_sequence_dynamic_inputs)
{
    recurrent_sequence_parameters param;

    param.batch_size = Dimension::dynamic();
    param.input_size = Dimension::dynamic();
    param.hidden_size = Dimension::dynamic();
    param.num_directions = Dimension::dynamic();
    param.seq_length = Dimension::dynamic();
    param.et = element::Type_t::f32;

    auto lstm_sequence = lstm_seq_tensor_initialization(param);
    lstm_sequence->validate_and_infer_types();

    EXPECT_EQ(lstm_sequence->get_output_partial_shape(0),
              (PartialShape{
                  param.batch_size, param.num_directions, param.seq_length, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(1),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_partial_shape(2),
              (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
    EXPECT_EQ(lstm_sequence->get_output_element_type(0), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(1), param.et);
    EXPECT_EQ(lstm_sequence->get_output_element_type(2), param.et);
}

TEST(type_prop, lstm_sequence_invalid_input_dimension)
{
    recurrent_sequence_parameters param;

    param.batch_size = 24;
    param.num_directions = 2;
    param.seq_length = 12;
    param.input_size = 8;
    param.hidden_size = 256;
    param.et = element::Type_t::f32;

    auto lstm_sequence = lstm_seq_tensor_initialization(param);
    auto invalid_rank0_tensor = make_shared<opset5::Parameter>(param.et, PartialShape{});

    // Validate invalid rank0 tensor for all inputs: X, initial_hidden_state, initial_cell_state W,
    // R, B
    for (auto i = 0; i < lstm_sequence->get_input_size(); i++)
    {
        lstm_sequence = lstm_seq_tensor_initialization(param);
        lstm_sequence->set_argument(i, invalid_rank0_tensor);
        ASSERT_THROW(lstm_sequence->validate_and_infer_types(), ngraph::CheckFailure)
            << "LSTMSequence node was created with invalid data.";
    }
}

TEST(type_prop, lstm_sequence_invalid_input_dynamic_rank)
{
    recurrent_sequence_parameters param;

    param.batch_size = 24;
    param.num_directions = 2;
    param.seq_length = 12;
    param.input_size = 8;
    param.hidden_size = 256;
    param.et = element::Type_t::f32;

    auto check_dynamic_lstm = [](const shared_ptr<opset5::LSTMSequence>& lstm) -> bool {
        return lstm->output(0).get_partial_shape() == PartialShape::dynamic() &&
               lstm->output(1).get_partial_shape() == PartialShape::dynamic() &&
               lstm->output(2).get_partial_shape() == PartialShape::dynamic() &&
               lstm->output(0).get_element_type() == lstm->input(0).get_element_type();
    };

    auto lstm_sequence = lstm_seq_tensor_initialization(param);
    auto invalid_dynamic_tensor =
        make_shared<opset5::Parameter>(param.et, PartialShape::dynamic(Rank::dynamic()));

    // Validate invalid dynamic tensor for all inputs: X, initial_hidden_state, initial_cell_state
    // W, R, B
    for (auto i = 0; i < lstm_sequence->get_input_size(); i++)
    {
        lstm_sequence = lstm_seq_tensor_initialization(param);
        lstm_sequence->set_argument(i, invalid_dynamic_tensor);
        lstm_sequence->validate_and_infer_types();
        EXPECT_EQ(check_dynamic_lstm(lstm_sequence), true);
    }
}
