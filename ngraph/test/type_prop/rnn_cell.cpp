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

TEST(type_prop, rnn_cell)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W = make_shared<op::Parameter>(element::f32, Shape{hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, hidden_size});

    const auto rnn_cell = make_shared<op::RNNCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(rnn_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(rnn_cell->get_output_shape(0), (Shape{batch_size, hidden_size}));
}

TEST(type_prop, rnn_cell_invalid_input)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;

    auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    auto R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});

    // Invalid W tensor shape.
    auto W = make_shared<op::Parameter>(element::f32, Shape{2 * hidden_size, input_size});
    try
    {
        const auto rnn_cell = make_shared<op::RNNCell>(X, H_t, W, R, hidden_size);
        FAIL() << "RNNCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor W must have shape"));
    }

    // Invalid R tensor shape.
    W = make_shared<op::Parameter>(element::f32, Shape{hidden_size, input_size});
    R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, 1});
    try
    {
        const auto rnn_cell = make_shared<op::RNNCell>(X, H_t, W, R, hidden_size);
        FAIL() << "RNNCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor R must have shape"));
    }

    // Invalid H_t tensor shape.
    R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    H_t = make_shared<op::Parameter>(element::f32, Shape{4, hidden_size});
    try
    {
        const auto rnn_cell = make_shared<op::RNNCell>(X, H_t, W, R, hidden_size);
        FAIL() << "RNNCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input tensor initial_hidden_state must have shape"));
    }

    // Invalid B tensor shape.
    H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2 * hidden_size});
    try
    {
        const auto rnn_cell = make_shared<op::RNNCell>(X, H_t, W, R, B, hidden_size);
        FAIL() << "RNNCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor B must have shape"));
    }
}
