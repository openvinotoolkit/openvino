// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset4.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, rnn_cell) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;

    const auto X = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto H_t = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W = make_shared<opset4::Parameter>(element::f32, Shape{hidden_size, input_size});
    const auto R = make_shared<opset4::Parameter>(element::f32, Shape{hidden_size, hidden_size});

    const auto rnn_cell = make_shared<opset4::RNNCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(rnn_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(rnn_cell->get_output_shape(0), (Shape{batch_size, hidden_size}));
}

TEST(type_prop, rnn_cell_with_bias) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;

    const auto X = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto H_t = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W = make_shared<opset4::Parameter>(element::f32, Shape{hidden_size, input_size});
    const auto R = make_shared<opset4::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    const auto B = make_shared<opset4::Parameter>(element::f32, Shape{hidden_size});

    const auto rnn_cell = make_shared<opset4::RNNCell>(X, H_t, W, R, B, hidden_size);
    EXPECT_EQ(rnn_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(rnn_cell->get_output_shape(0), (Shape{batch_size, hidden_size}));
}

TEST(type_prop, rnn_cell_invalid_input) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;

    auto X = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, input_size});
    auto R = make_shared<opset4::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    auto H_t = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, hidden_size});

    // Invalid W tensor shape.
    auto W = make_shared<opset4::Parameter>(element::f32, Shape{2 * hidden_size, input_size});
    try {
        const auto rnn_cell = make_shared<opset4::RNNCell>(X, H_t, W, R, hidden_size);
        FAIL() << "RNNCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("First dimension of W input shape is required to be compatible with 3. Got shape: 6."));
    }

    // Invalid R tensor shape.
    W = make_shared<opset4::Parameter>(element::f32, Shape{hidden_size, input_size});
    R = make_shared<opset4::Parameter>(element::f32, Shape{hidden_size, 1});
    try {
        const auto rnn_cell = make_shared<opset4::RNNCell>(X, H_t, W, R, hidden_size);
        FAIL() << "RNNCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Dimension `hidden_size` is not matched between inputs"));
    }

    // Invalid H_t tensor shape.
    R = make_shared<opset4::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    H_t = make_shared<opset4::Parameter>(element::f32, Shape{4, hidden_size});
    try {
        const auto rnn_cell = make_shared<opset4::RNNCell>(X, H_t, W, R, hidden_size);
        FAIL() << "RNNCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Dimension `batch_size` is not matched between inputs"));
    }

    // Invalid B tensor shape.
    H_t = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, hidden_size});
    auto B = make_shared<opset4::Parameter>(element::f32, Shape{2 * hidden_size});
    try {
        const auto rnn_cell = make_shared<opset4::RNNCell>(X, H_t, W, R, B, hidden_size);
        FAIL() << "RNNCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("First dimension of B input shape is required to be compatible with 3. Got shape: 6."));
    }
}

TEST(type_prop, rnn_cell_dynamic_batch_size) {
    const auto batch_size = Dimension::dynamic();
    const size_t input_size = 3;
    const size_t hidden_size = 3;

    const auto X = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, input_size});
    const auto H_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    const auto W = make_shared<opset4::Parameter>(element::f32, PartialShape{hidden_size, input_size});
    const auto R = make_shared<opset4::Parameter>(element::f32, PartialShape{hidden_size, hidden_size});

    const auto rnn_cell = make_shared<opset4::RNNCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(rnn_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(rnn_cell->get_output_partial_shape(0), (PartialShape{batch_size, hidden_size}));
}

TEST(type_prop, rnn_cell_dynamic_hidden_size) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const auto hidden_size = Dimension::dynamic();

    const auto X = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, input_size});
    const auto H_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    const auto W = make_shared<opset4::Parameter>(element::f32, PartialShape{hidden_size, input_size});
    const auto R = make_shared<opset4::Parameter>(element::f32, PartialShape{hidden_size, hidden_size});

    const auto rnn_cell = make_shared<opset4::RNNCell>(X, H_t, W, R, 3);
    EXPECT_EQ(rnn_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(rnn_cell->get_output_partial_shape(0), (PartialShape{batch_size, hidden_size}));
}

TEST(type_prop, rnn_cell_dynamic_inputs) {
    const auto batch_size = Dimension::dynamic();
    const auto input_size = Dimension::dynamic();
    const auto hidden_size = Dimension::dynamic();

    const auto X = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, input_size});
    const auto R = make_shared<opset4::Parameter>(element::f32, PartialShape{hidden_size, hidden_size});
    const auto W = make_shared<opset4::Parameter>(element::f32, PartialShape{hidden_size, input_size});
    const auto H_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});

    const auto rnn_cell = make_shared<opset4::RNNCell>(X, H_t, W, R, 2);

    EXPECT_EQ(rnn_cell->get_output_partial_shape(0), (PartialShape{batch_size, hidden_size}));
    EXPECT_EQ(rnn_cell->get_output_element_type(0), element::f32);
}

TEST(type_prop, rnn_cell_invalid_input_rank0) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;

    auto X = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, input_size});
    auto R = make_shared<opset4::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    auto H_t = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, hidden_size});

    // Invalid rank0 for W tensor.
    auto W = make_shared<opset4::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<opset4::RNNCell>(X, H_t, W, R, hidden_size), ov::NodeValidationFailure)
        << "RNNCell node was created with invalid data.";

    // Invalid rank0 for X tensor.
    W = make_shared<opset4::Parameter>(element::f32, Shape{hidden_size, input_size});
    X = make_shared<opset4::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<opset4::RNNCell>(X, H_t, W, R, hidden_size), ov::NodeValidationFailure)
        << "RNNCell node was created with invalid data.";

    // Invalid rank0 for H_t tensor.
    X = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, input_size});
    H_t = make_shared<opset4::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<opset4::RNNCell>(X, H_t, W, R, hidden_size), ov::NodeValidationFailure)
        << "RNNCell node was created with invalid data.";

    // Invalid rank0 for R tensor.
    H_t = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, hidden_size});
    R = make_shared<opset4::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<opset4::RNNCell>(X, H_t, W, R, hidden_size), ov::NodeValidationFailure)
        << "RNNCell node was created with invalid data.";

    // Invalid rank0 for B tensor.
    R = make_shared<opset4::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    auto B = make_shared<opset4::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<opset4::RNNCell>(X, H_t, W, R, B, hidden_size),
                 ov::NodeValidationFailure)
        << "RNNCell node was created with invalid data.";
}

TEST(type_prop, rnn_cell_input_dynamic_rank) {
    const int64_t batch_size = 2;
    const int64_t input_size = 3;
    const size_t hidden_size = 3;
    const auto hidden_size_dim = Dimension(static_cast<int64_t>(hidden_size));

    auto X = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, input_size});
    auto R = make_shared<opset4::Parameter>(element::f32, PartialShape{hidden_size_dim, hidden_size_dim});
    auto H_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});

    auto check_dynamic_rnn = [=](const shared_ptr<opset4::RNNCell>& rnn) -> bool {
        return rnn->output(0).get_partial_shape() == PartialShape{batch_size, hidden_size_dim} &&
               rnn->output(0).get_element_type() == rnn->input(0).get_element_type();
    };
    // Invalid dynamic rank for W tensor.
    auto W = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto rnn_w = make_shared<opset4::RNNCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(check_dynamic_rnn(rnn_w), true);

    // Invalid dynamic rank for X tensor.
    W = make_shared<opset4::Parameter>(element::f32, PartialShape{hidden_size_dim, input_size});
    X = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto rnn_x = make_shared<opset4::RNNCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(check_dynamic_rnn(rnn_x), true);

    // Invalid dynamic rank for H_t tensor.
    X = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, input_size});
    H_t = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto rnn_h = make_shared<opset4::RNNCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(check_dynamic_rnn(rnn_h), true);

    // Invalid dynamic rank for R tensor.
    H_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    R = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto rnn_r = make_shared<opset4::RNNCell>(X, H_t, W, R, hidden_size);
    EXPECT_EQ(check_dynamic_rnn(rnn_r), true);

    // Invalid dynamic rank for B tensor.
    R = make_shared<opset4::Parameter>(element::f32, PartialShape{hidden_size_dim, hidden_size_dim});
    auto B = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto rnn_b = make_shared<opset4::RNNCell>(X, H_t, W, R, B, hidden_size);
    EXPECT_EQ(check_dynamic_rnn(rnn_b), true);
}
