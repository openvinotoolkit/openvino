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

TEST(type_prop, lstm_cell)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});

    const auto lstm_cell = make_shared<op::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
    EXPECT_EQ(lstm_cell->get_hidden_size(), hidden_size);
    EXPECT_EQ(lstm_cell->get_clip(), 0.f);
    EXPECT_TRUE(lstm_cell->get_activations_alpha().empty());
    EXPECT_TRUE(lstm_cell->get_activations_beta().empty());
    EXPECT_EQ(lstm_cell->get_activations()[0], "sigmoid");
    EXPECT_EQ(lstm_cell->get_activations()[1], "tanh");
    EXPECT_EQ(lstm_cell->get_activations()[2], "tanh");
    EXPECT_EQ(lstm_cell->get_weights_format(), op::LSTMWeightsFormat::IFCO);
    EXPECT_FALSE(lstm_cell->get_input_forget());
    EXPECT_EQ(lstm_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(lstm_cell->get_output_shape(0), (Shape{batch_size, hidden_size}));
    EXPECT_EQ(lstm_cell->get_output_element_type(1), element::f32);
    EXPECT_EQ(lstm_cell->get_output_shape(1), (Shape{batch_size, hidden_size}));
}

TEST(type_prop, lstm_cell_invalid_input)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    auto C_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});

    // Invalid W tensor shape.
    auto W = make_shared<op::Parameter>(element::f32, Shape{1 * hidden_size, input_size});
    try
    {
        const auto lstm_cell = make_shared<op::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
        FAIL() << "LSTMCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor W must have shape"));
    }

    // Invalid R tensor shape.
    W = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    R = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, 1});
    try
    {
        const auto lstm_cell = make_shared<op::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
        FAIL() << "LSTMCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor R must have shape"));
    }

    // Invalid H_t tensor shape.
    R = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    H_t = make_shared<op::Parameter>(element::f32, Shape{4, hidden_size});
    try
    {
        const auto lstm_cell = make_shared<op::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
        FAIL() << "LSTMCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input tensor initial_hidden_state must have shape"));
    }

    // Invalid C_t tensor shape.
    H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    C_t = make_shared<op::Parameter>(element::f32, Shape{4, hidden_size});
    try
    {
        const auto lstm_cell = make_shared<op::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
        FAIL() << "LSTMCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input tensor initial_cell_state must have shape"));
    }

    // Invalid B tensor shape.
    C_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2 * gates_count * hidden_size});
    auto P = make_shared<op::Parameter>(element::f32, Shape{3 * hidden_size});
    try
    {
        const auto lstm_cell = make_shared<op::LSTMCell>(X, H_t, C_t, W, R, B, P, hidden_size);
        FAIL() << "LSTMCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor B must have shape"));
    }

    // Invalid P tensor shape.
    B = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size});
    P = make_shared<op::Parameter>(element::f32, Shape{hidden_size});
    try
    {
        const auto lstm_cell = make_shared<op::LSTMCell>(X, H_t, C_t, W, R, B, P, hidden_size);
        FAIL() << "LSTMCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor P must have shape"));
    }
}
