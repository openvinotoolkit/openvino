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
#include "ngraph/opsets/opset4.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, gru_cell)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});

    const auto gru_cell = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(gru_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(gru_cell->get_output_shape(0), (Shape{batch_size, hidden_size}));
}

TEST(type_prop, gru_cell_invalid_input)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});

    // Invalid W tensor shape.
    auto W = make_shared<op::Parameter>(element::f32, Shape{hidden_size, input_size});
    try
    {
        const auto gru_cell = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size);
        FAIL() << "GRUCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Parameter hidden_size mistmatched in W input."));
    }

    // Invalid R tensor shape.
    W = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, 1});
    try
    {
        const auto gru_cell = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size);
        FAIL() << "GRUCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Parameter hidden_size not matched for R and initial_hidden_state inputs."));
    }

    // Invalid H_t tensor shape.
    R = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    H_t = make_shared<op::Parameter>(element::f32, Shape{4, hidden_size});
    try
    {
        const auto gru_cell = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size);
        FAIL() << "GRUCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Parameter batch_size not matched for X and initial_hidden_state inputs."));
    }

    // Invalid B tensor shape.
    H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    auto B = make_shared<op::Parameter>(element::f32, Shape{hidden_size});
    try
    {
        const auto gru_cell = make_shared<opset4::GRUCell>(X, H_t, W, R, B, hidden_size);
        FAIL() << "GRUCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Parameter hidden_size mistmatched in B input."));
    }
}

TEST(type_prop, gru_cell_dynamic_batch_size)
{
    const auto batch_size = Dimension::dynamic();
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    const auto X = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, input_size});
    const auto W = make_shared<op::Parameter>(element::f32,
                                              PartialShape{gates_count * hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32,
                                              PartialShape{gates_count * hidden_size, hidden_size});
    const auto H_t =
        make_shared<op::Parameter>(element::f32, PartialShape{batch_size, hidden_size});

    const auto gru_cell = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(gru_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(gru_cell->get_output_partial_shape(0), (PartialShape{batch_size, hidden_size}));
}

TEST(type_prop, gru_cell_dynamic_hidden_size)
{
    const auto batch_size = 2;
    const size_t input_size = 3;
    const auto hidden_size = Dimension::dynamic();
    const size_t gates_count = 3;

    const auto X = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, input_size});
    const auto W = make_shared<op::Parameter>(element::f32,
                                              PartialShape{hidden_size * gates_count, input_size});
    const auto R = make_shared<op::Parameter>(element::f32,
                                              PartialShape{hidden_size * gates_count, hidden_size});
    const auto H_t =
        make_shared<op::Parameter>(element::f32, PartialShape{batch_size, hidden_size});

    const auto gru_cell = make_shared<opset4::GRUCell>(X, H_t, W, R, 3);
    EXPECT_EQ(gru_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(gru_cell->get_output_partial_shape(0), (PartialShape{batch_size, hidden_size}));
}

TEST(type_prop, gru_cell_dynamic_inputs)
{
    const auto batch_size = Dimension::dynamic();
    const auto input_size = Dimension::dynamic();
    const auto hidden_size = Dimension::dynamic();

    const auto X = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, input_size});
    const auto W = make_shared<op::Parameter>(element::f32, PartialShape{hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32, PartialShape{hidden_size, hidden_size});
    const auto H_t =
        make_shared<op::Parameter>(element::f32, PartialShape{batch_size, hidden_size});

    const auto gru_cell = make_shared<opset4::GRUCell>(X, H_t, W, R, 2);

    EXPECT_EQ(gru_cell->get_output_partial_shape(0), (PartialShape{batch_size, hidden_size}));
    EXPECT_EQ(gru_cell->get_output_element_type(0), element::f32);
}

TEST(type_prop, gru_cell_invalid_input_rank0)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    auto X = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, input_size});
    auto R = make_shared<op::Parameter>(element::f32,
                                        PartialShape{gates_count * hidden_size, hidden_size});
    auto H_t = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, hidden_size});

    // Invalid rank0 for W tensor.
    auto W = make_shared<op::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size),
                 ngraph::NodeValidationFailure)
        << "GRUCell node was created with invalid data.";

    // Invalid rank0 for X tensor.
    W = make_shared<op::Parameter>(element::f32,
                                   PartialShape{gates_count * hidden_size, input_size});
    X = make_shared<op::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size),
                 ngraph::NodeValidationFailure)
        << "GRUCell node was created with invalid data.";

    // Invalid rank0 for H_t tensor.
    X = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, input_size});
    H_t = make_shared<op::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size),
                 ngraph::NodeValidationFailure)
        << "GRUCell node was created with invalid data.";

    // Invalid rank0 for R tensor.
    H_t = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    R = make_shared<op::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size),
                 ngraph::NodeValidationFailure)
        << "GRUCell node was created with invalid data.";

    // Invalid rank0 for B tensor.
    R = make_shared<op::Parameter>(element::f32,
                                   PartialShape{gates_count * hidden_size, input_size});
    auto B = make_shared<op::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(make_shared<opset4::GRUCell>(X, H_t, W, R, B, hidden_size),
                 ngraph::NodeValidationFailure)
        << "GRUCell node was created with invalid data.";
}

TEST(type_prop, gru_cell_invalid_input_dynamic_rank)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    auto X = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, input_size});
    auto R = make_shared<op::Parameter>(element::f32,
                                        PartialShape{gates_count * hidden_size, hidden_size});
    auto H_t = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, hidden_size});

    auto check_dynamic_gru = [](const shared_ptr<opset4::GRUCell>& gru) -> bool {
        return gru->output(0).get_partial_shape() == PartialShape::dynamic() &&
               gru->output(0).get_element_type() == gru->input(0).get_element_type();
    };

    // Invalid dynamic rank for W tensor.
    auto W = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto gru_w = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(check_dynamic_gru(gru_w), true);

    // Invalid dynamic rank for X tensor.
    W = make_shared<op::Parameter>(element::f32, PartialShape{hidden_size, input_size});
    X = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto gru_x = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(check_dynamic_gru(gru_x), true);

    // Invalid dynamic rank for H_t tensor.
    X = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, input_size});
    H_t = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto gru_h = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(check_dynamic_gru(gru_h), true);

    // Invalid dynamic rank for R tensor.
    H_t = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    R = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto gru_r = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(check_dynamic_gru(gru_r), true);

    // Invalid dynamic rank for B tensor.
    R = make_shared<op::Parameter>(element::f32,
                                   PartialShape{gates_count * hidden_size, hidden_size});
    auto B = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto gru_b = make_shared<opset4::GRUCell>(X, H_t, W, R, B, hidden_size);
    EXPECT_EQ(check_dynamic_gru(gru_b), true);
}
