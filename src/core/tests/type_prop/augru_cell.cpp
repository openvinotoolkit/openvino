// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "augru_cell.hpp"

#include "gtest/gtest.h"
#include "openvino/opsets/opset9.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, augru_cell) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    const auto X = make_shared<opset9::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto H_t = make_shared<opset9::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto B = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size});
    const auto A = make_shared<opset9::Parameter>(element::f32, Shape{batch_size, 1});

    const auto augru_cell = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
    EXPECT_EQ(augru_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(augru_cell->get_output_shape(0), (Shape{batch_size, hidden_size}));
}

TEST(type_prop, augru_cell_invalid_input) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;
    auto B = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size});
    auto A = make_shared<opset9::Parameter>(element::f32, Shape{batch_size, 1});

    const auto X = make_shared<opset9::Parameter>(element::f32, Shape{batch_size, input_size});
    auto R = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    auto H_t = make_shared<opset9::Parameter>(element::f32, Shape{batch_size, hidden_size});

    // Invalid W tensor shape.
    auto W = make_shared<opset9::Parameter>(element::f32, Shape{hidden_size, input_size});
    try {
        const auto gru_cell = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
        FAIL() << "AUGRUCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Parameter hidden_size mistmatched in W input."));
    }

    // Invalid R tensor shape.
    W = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    R = make_shared<opset9::Parameter>(element::f32, Shape{hidden_size, 1});
    try {
        const auto gru_cell = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
        FAIL() << "AUGRUCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Dimension hidden_size not matched for R and initial_hidden_state inputs."));
    }

    // Invalid H_t tensor shape.
    R = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    H_t = make_shared<opset9::Parameter>(element::f32, Shape{4, hidden_size});
    try {
        const auto gru_cell = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
        FAIL() << "AUGRUCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Dimension batch_size is not matched between inputs."));
    }

    // Invalid B tensor shape.
    H_t = make_shared<opset9::Parameter>(element::f32, Shape{batch_size, hidden_size});
    B = make_shared<opset9::Parameter>(element::f32, Shape{hidden_size});
    try {
        const auto gru_cell = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
        FAIL() << "GRUCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Parameter hidden_size mistmatched in B input. Current value is: 3, expected: 9."));
    }

    // Invalid A tensor shape.
    A = make_shared<opset9::Parameter>(element::f32, Shape{hidden_size});
    B = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size});
    try {
        const auto gru_cell = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
        FAIL() << "GRUCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'A' input must be a 2D tensor."));
    }
}

TEST(type_prop, augru_cell_dynamic_batch_size) {
    const auto batch_size = Dimension::dynamic();
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    const auto X = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, input_size});
    const auto W = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
    const auto R = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    const auto B = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size});
    const auto A = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, 1});

    const auto augru_cell = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
    EXPECT_EQ(augru_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(augru_cell->get_output_partial_shape(0), (PartialShape{batch_size, hidden_size}));
}

TEST(type_prop, augru_cell_dynamic_batch_and_input_size) {
    const auto batch_size = Dimension::dynamic();
    const auto input_size = Dimension::dynamic();
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    const auto X = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, input_size});
    const auto W = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
    const auto R = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    const auto B = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size});
    const auto A = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, 1});

    const auto augru_cell = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
    EXPECT_EQ(augru_cell->get_output_partial_shape(0), (PartialShape{batch_size, hidden_size}));
    EXPECT_EQ(augru_cell->get_output_element_type(0), element::f32);
}

TEST(type_prop, augru_cell_invalid_input_rank) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    auto X = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, input_size});
    auto R = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    auto H_t = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    auto B = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size});
    auto A = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, 1});

    // Invalid rank for W tensor.
    auto W = make_shared<opset9::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size),
                 ngraph::NodeValidationFailure)
        << "AUGRUCell node was created with invalid data.";

    // Invalid rank for X tensor.
    W = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
    X = make_shared<opset9::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size),
                 ngraph::NodeValidationFailure)
        << "AUGRUCell node was created with invalid data.";

    // Invalid rank for H_t tensor.
    X = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, input_size});
    H_t = make_shared<opset9::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size),
                 ngraph::NodeValidationFailure)
        << "AUGRUCell node was created with invalid data.";

    // Invalid rank for R tensor.
    H_t = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    R = make_shared<opset9::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size),
                 ngraph::NodeValidationFailure)
        << "AUGRUCell node was created with invalid data.";

    // Invalid rank for B tensor.
    R = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
    B = make_shared<opset9::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size),
                 ngraph::NodeValidationFailure)
        << "AUGRUCell node was created with invalid data.";

    // Invalid rank for A tensor.
    B = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size});
    A = make_shared<opset9::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size),
                 ngraph::NodeValidationFailure)
        << "AUGRUCell node was created with invalid data.";
}

TEST(type_prop, augru_cell_invalid_input_dynamic_rank) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    auto X = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, input_size});
    auto R = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    auto H_t = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    auto B = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size});
    auto A = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, 1});

    auto check_dynamic_gru = [](const shared_ptr<op::v0::AUGRUCell>& augru) -> bool {
        return augru->output(0).get_partial_shape() == PartialShape::dynamic(2) &&
               augru->output(0).get_element_type() == augru->input(0).get_element_type();
    };

    // Invalid dynamic rank for W tensor.
    auto W = make_shared<opset9::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto augru_w = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
    EXPECT_TRUE(check_dynamic_gru(augru_w));

    // Invalid dynamic rank for X tensor.
    W = make_shared<opset9::Parameter>(element::f32, PartialShape{hidden_size, input_size});
    X = make_shared<opset9::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto augru_x = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
    EXPECT_TRUE(check_dynamic_gru(augru_x));

    // Invalid dynamic rank for H_t tensor.
    X = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, input_size});
    H_t = make_shared<opset9::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto augru_h = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
    EXPECT_TRUE(check_dynamic_gru(augru_h));

    // Invalid dynamic rank for R tensor.
    H_t = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    R = make_shared<opset9::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto augru_r = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
    EXPECT_TRUE(check_dynamic_gru(augru_r));

    // Invalid dynamic rank for B tensor.
    R = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    B = make_shared<opset9::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto augru_b = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
    EXPECT_TRUE(check_dynamic_gru(augru_b));

    // Invalid dynamic rank for A tensor.
    B = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size});
    A = make_shared<opset9::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto augru_a = make_shared<op::v0::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
    EXPECT_TRUE(check_dynamic_gru(augru_a));
}
