// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;
using namespace testing;

TEST(type_prop, gru_cell) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});

    const auto gru_cell = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(gru_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(gru_cell->get_output_shape(0), (Shape{batch_size, hidden_size}));
}

TEST(type_prop, gru_cell_invalid_input) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    auto R = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});

    // Invalid W tensor shape.
    auto W = make_shared<op::Parameter>(element::f32, Shape{hidden_size, input_size});
    OV_EXPECT_THROW(auto op = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size),
                    ov::NodeValidationFailure,
                    HasSubstr("First dimension of W input shape is required to be compatible"));

    // Invalid R tensor shape.
    W = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, 1});
    OV_EXPECT_THROW(auto op = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size),
                    ov::NodeValidationFailure,
                    HasSubstr("Dimension `hidden_size` is not matched between inputs"));

    // Invalid H_t tensor shape.
    R = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    H_t = make_shared<op::Parameter>(element::f32, Shape{4, hidden_size});
    OV_EXPECT_THROW(auto op = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size),
                    ov::NodeValidationFailure,
                    HasSubstr("Dimension `batch_size` is not matched between inputs"));

    // Invalid B tensor shape.
    H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    auto B = make_shared<op::Parameter>(element::f32, Shape{hidden_size});
    OV_EXPECT_THROW(auto op = make_shared<opset4::GRUCell>(X, H_t, W, R, B, hidden_size),
                    ov::NodeValidationFailure,
                    HasSubstr("First dimension of B input shape is required to be compatible"));
}

TEST(type_prop, gru_cell_dynamic_batch_size) {
    const auto batch_size = Dimension::dynamic();
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    const auto X = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, input_size});
    const auto W = make_shared<op::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, hidden_size});

    const auto gru_cell = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(gru_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(gru_cell->get_output_partial_shape(0), (PartialShape{batch_size, hidden_size}));
}

TEST(type_prop, gru_cell_dynamic_hidden_size) {
    const auto batch_size = 2;
    const size_t input_size = 3;
    const auto hidden_size = Dimension::dynamic();
    const size_t gates_count = 3;

    const auto X = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, input_size});
    const auto W = make_shared<op::Parameter>(element::f32, PartialShape{hidden_size * gates_count, input_size});
    const auto R = make_shared<op::Parameter>(element::f32, PartialShape{hidden_size * gates_count, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, hidden_size});

    const auto gru_cell = make_shared<opset4::GRUCell>(X, H_t, W, R, 3);
    EXPECT_EQ(gru_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(gru_cell->get_output_partial_shape(0), (PartialShape{batch_size, hidden_size}));
}

TEST(type_prop, gru_cell_dynamic_inputs) {
    const auto batch_size = Dimension::dynamic();
    const auto input_size = Dimension::dynamic();
    const auto hidden_size = Dimension::dynamic();

    const auto X = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, input_size});
    const auto W = make_shared<op::Parameter>(element::f32, PartialShape{hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32, PartialShape{hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, hidden_size});

    const auto gru_cell = make_shared<opset4::GRUCell>(X, H_t, W, R, 2);

    EXPECT_EQ(gru_cell->get_output_partial_shape(0), (PartialShape{batch_size, hidden_size}));
    EXPECT_EQ(gru_cell->get_output_element_type(0), element::f32);
}

TEST(type_prop, gru_cell_invalid_input_rank0) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    auto X = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, input_size});
    auto R = make_shared<op::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    auto H_t = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, hidden_size});

    // Invalid rank0 for W tensor.
    auto W = make_shared<op::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size),
                 ngraph::NodeValidationFailure)
        << "GRUCell node was created with invalid data.";

    // Invalid rank0 for X tensor.
    W = make_shared<op::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
    X = make_shared<op::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size),
                 ngraph::NodeValidationFailure)
        << "GRUCell node was created with invalid data.";

    // Invalid rank0 for H_t tensor.
    X = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, input_size});
    H_t = make_shared<op::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size),
                 ngraph::NodeValidationFailure)
        << "GRUCell node was created with invalid data.";

    // Invalid rank0 for R tensor.
    H_t = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    R = make_shared<op::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size),
                 ngraph::NodeValidationFailure)
        << "GRUCell node was created with invalid data.";

    // Invalid rank0 for B tensor.
    R = make_shared<op::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
    auto B = make_shared<op::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<opset4::GRUCell>(X, H_t, W, R, B, hidden_size),
                 ngraph::NodeValidationFailure)
        << "GRUCell node was created with invalid data.";
}

TEST(type_prop, gru_cell_input_dynamic_rank) {
    int64_t batch_size = 2;
    int64_t input_size = 3;
    int64_t hidden_size = 3;
    int64_t gates_count = 3;

    auto X = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, input_size});
    auto R = make_shared<op::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    auto H_t = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, hidden_size});

    auto check_dynamic_gru = [&](const shared_ptr<opset4::GRUCell>& gru) -> bool {
        return gru->output(0).get_partial_shape() == PartialShape{batch_size, hidden_size} &&
               gru->output(0).get_element_type() == gru->input(0).get_element_type();
    };

    // Dynamic rank for W tensor.
    auto W = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto gru_w = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(check_dynamic_gru(gru_w), true);

    // Dynamic rank for X tensor.
    W = make_shared<op::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
    X = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto gru_x = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(check_dynamic_gru(gru_x), true);

    // Dynamic rank for H_t tensor.
    X = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, input_size});
    H_t = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto gru_h = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(check_dynamic_gru(gru_h), true);

    // Dynamic rank for R tensor.
    H_t = make_shared<op::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    R = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto gru_r = make_shared<opset4::GRUCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(check_dynamic_gru(gru_r), true);

    // Dynamic rank for B tensor.
    R = make_shared<op::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    auto B = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto gru_b = make_shared<opset4::GRUCell>(X, H_t, W, R, B, hidden_size);
    EXPECT_EQ(check_dynamic_gru(gru_b), true);
}
