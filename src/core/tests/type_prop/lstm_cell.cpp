// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset4.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, lstm_cell) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    const auto X = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W = make_shared<opset4::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R = make_shared<opset4::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, hidden_size});

    const auto lstm_cell = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
    EXPECT_EQ(lstm_cell->get_hidden_size(), hidden_size);
    EXPECT_EQ(lstm_cell->get_clip(), 0.f);
    EXPECT_TRUE(lstm_cell->get_activations_alpha().empty());
    EXPECT_TRUE(lstm_cell->get_activations_beta().empty());
    EXPECT_EQ(lstm_cell->get_activations()[0], "sigmoid");
    EXPECT_EQ(lstm_cell->get_activations()[1], "tanh");
    EXPECT_EQ(lstm_cell->get_activations()[2], "tanh");
    EXPECT_EQ(lstm_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(lstm_cell->get_output_shape(0), (Shape{batch_size, hidden_size}));
    EXPECT_EQ(lstm_cell->get_output_element_type(1), element::f32);
    EXPECT_EQ(lstm_cell->get_output_shape(1), (Shape{batch_size, hidden_size}));
}

TEST(type_prop, lstm_cell_invalid_input) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    auto X = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, input_size});
    auto R = make_shared<opset4::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    auto H_t = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, hidden_size});
    auto C_t = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, hidden_size});

    // Invalid W tensor shape.
    auto W = make_shared<opset4::Parameter>(element::f32, Shape{1 * hidden_size, input_size});
    try {
        const auto lstm_cell = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
        FAIL() << "LSTMCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("First dimension of W input shape is required to be compatible"));
    }

    // Invalid R tensor shape.
    W = make_shared<opset4::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    R = make_shared<opset4::Parameter>(element::f32, Shape{gates_count * hidden_size, 1});
    try {
        const auto lstm_cell = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
        FAIL() << "LSTMCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Dimension `hidden_size` is not matched between inputs"));
    }

    // Invalid H_t tensor shape.
    R = make_shared<opset4::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    H_t = make_shared<opset4::Parameter>(element::f32, Shape{4, hidden_size});
    try {
        const auto lstm_cell = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
        FAIL() << "LSTMCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Dimension `batch_size` is not matched between inputs"));
    }

    // Invalid C_t tensor shape.
    H_t = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, hidden_size});
    C_t = make_shared<opset4::Parameter>(element::f32, Shape{4, hidden_size});
    try {
        const auto lstm_cell = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
        FAIL() << "LSTMCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Dimension `batch_size` is not matched between inputs"));
    }

    // Invalid B tensor shape.
    C_t = make_shared<opset4::Parameter>(element::f32, Shape{batch_size, hidden_size});
    auto B = make_shared<opset4::Parameter>(element::f32, Shape{2 * gates_count * hidden_size});
    try {
        const auto lstm_cell = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, B, hidden_size);
        FAIL() << "LSTMCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("First dimension of B input shape is required to be compatible with 12. Got shape: 24."));
    }
}

TEST(type_prop, lstm_cell_dynamic_batch_size) {
    const auto batch_size = Dimension::dynamic();
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    const auto X = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, input_size});
    const auto W = make_shared<opset4::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
    const auto R = make_shared<opset4::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    const auto C_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});

    const auto lstm_cell = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size);

    EXPECT_EQ(lstm_cell->get_output_partial_shape(0), (PartialShape{batch_size, hidden_size}));
    EXPECT_EQ(lstm_cell->get_output_partial_shape(1), (PartialShape{batch_size, hidden_size}));
    EXPECT_EQ(lstm_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(lstm_cell->get_output_element_type(1), element::f32);
}

TEST(type_prop, lstm_cell_dynamic_hidden_size) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const auto hidden_size = Dimension::dynamic();
    const size_t gates_count = 4;

    const auto X = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, input_size});
    const auto W = make_shared<opset4::Parameter>(element::f32, PartialShape{hidden_size * gates_count, input_size});
    const auto R = make_shared<opset4::Parameter>(element::f32, PartialShape{hidden_size * gates_count, hidden_size});
    const auto H_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    const auto C_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});

    const auto lstm_cell = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, 3);

    EXPECT_EQ(lstm_cell->get_output_partial_shape(0), (PartialShape{batch_size, 3}));
    EXPECT_EQ(lstm_cell->get_output_partial_shape(1), (PartialShape{batch_size, 3}));
    EXPECT_EQ(lstm_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(lstm_cell->get_output_element_type(1), element::f32);
}

TEST(type_prop, lstm_cell_dynamic_inputs) {
    const auto batch_size = Dimension::dynamic();
    const auto input_size = Dimension::dynamic();
    const auto hidden_size = Dimension::dynamic();
    const size_t gates_count = 4;

    const auto X = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, input_size});
    const auto W = make_shared<opset4::Parameter>(element::f32, PartialShape{hidden_size * gates_count, input_size});
    const auto R = make_shared<opset4::Parameter>(element::f32, PartialShape{hidden_size * gates_count, hidden_size});
    const auto H_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    const auto C_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});

    const auto lstm_cell = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, 3);

    EXPECT_EQ(lstm_cell->get_output_partial_shape(0), (PartialShape{batch_size, 3}));
    EXPECT_EQ(lstm_cell->get_output_partial_shape(1), (PartialShape{batch_size, 3}));
    EXPECT_EQ(lstm_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(lstm_cell->get_output_element_type(1), element::f32);
}

TEST(type_prop, lstm_cell_invalid_input_rank0) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    auto X = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, input_size});
    auto W = make_shared<opset4::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
    auto R = make_shared<opset4::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    auto H_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    auto C_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});

    // Invalid rank0 for W tensor.
    W = make_shared<opset4::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size),
                 ov::NodeValidationFailure)
        << "LSTMCell node was created with invalid data.";

    // Invalid rank0 for X tensor.
    W = make_shared<opset4::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
    X = make_shared<opset4::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size),
                 ov::NodeValidationFailure)
        << "LSTMCell node was created with invalid data.";

    // Invalid rank0 for H_t tensor.
    X = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, input_size});
    H_t = make_shared<opset4::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size),
                 ov::NodeValidationFailure)
        << "LSTMCell node was created with invalid data.";

    // Invalid rank0 for C_t tensor.
    H_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    C_t = make_shared<opset4::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size),
                 ov::NodeValidationFailure)
        << "LSTMCell node was created with invalid data.";

    // Invalid rank0 for R tensor.
    C_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    R = make_shared<opset4::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size),
                 ov::NodeValidationFailure)
        << "LSTMCell node was created with invalid data.";

    // Invalid rank0 for B tensor.
    R = make_shared<opset4::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    auto B = make_shared<opset4::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, B, hidden_size),
                 ov::NodeValidationFailure)
        << "LSTMCell node was created with invalid data.";
}

TEST(type_prop, lstm_cell_invalid_input_dynamic_rank) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    auto X = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, input_size});
    auto W = make_shared<opset4::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
    auto R = make_shared<opset4::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    auto H_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    auto C_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});

    auto check_dynamic_lstm = [=](const shared_ptr<opset4::LSTMCell>& lstm) -> bool {
        const int64_t target_batch_size = batch_size;
        const int64_t target_hidden_size = hidden_size;
        return lstm->output(0).get_partial_shape() == PartialShape{target_batch_size, target_hidden_size} &&
               lstm->output(1).get_partial_shape() == PartialShape{target_batch_size, target_hidden_size} &&
               lstm->output(0).get_element_type() == lstm->input(0).get_element_type();
    };

    // Invalid dynamic rank for W tensor.
    W = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto lstm = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
    EXPECT_EQ(check_dynamic_lstm(lstm), true);

    // Invalid dynamic rank for X tensor.
    W = make_shared<opset4::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
    X = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    lstm = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
    EXPECT_EQ(check_dynamic_lstm(lstm), true);

    // Invalid dynamic rank for H_t tensor.
    X = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, input_size});
    H_t = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    lstm = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
    EXPECT_EQ(check_dynamic_lstm(lstm), true);

    // Invalid dynamic rank for C_t tensor.
    H_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    C_t = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    lstm = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
    EXPECT_EQ(check_dynamic_lstm(lstm), true);

    // Invalid dynamic rank for R tensor.
    C_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    R = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    lstm = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
    EXPECT_EQ(check_dynamic_lstm(lstm), true);

    // Invalid dynamic rank for B tensor.
    R = make_shared<opset4::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    auto B = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    lstm = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, B, hidden_size);
    EXPECT_EQ(check_dynamic_lstm(lstm), true);
}

TEST(type_prop, lstm_cell_shape_from_partial) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    auto check_dynamic_lstm = [=](const shared_ptr<opset4::LSTMCell>& lstm) -> bool {
        const int64_t target_batch_size = batch_size;
        const int64_t target_hidden_size = hidden_size;
        return lstm->output(0).get_partial_shape() == PartialShape{target_batch_size, target_hidden_size} &&
               lstm->output(1).get_partial_shape() == PartialShape{target_batch_size, target_hidden_size} &&
               lstm->output(0).get_element_type() == lstm->input(0).get_element_type();
    };
    {
        // from h & w
        auto X = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
        auto W = make_shared<opset4::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
        auto R = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
        auto H_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, -1});
        auto C_t = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
        auto lstm = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
        EXPECT_EQ(check_dynamic_lstm(lstm), true);
    }

    {
        // from x & w
        auto X = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, input_size});
        auto W = make_shared<opset4::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
        auto R = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
        auto H_t = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
        auto C_t = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
        auto lstm = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
        EXPECT_EQ(check_dynamic_lstm(lstm), true);
    }

    {
        // only valid rank for H_t tensor.
        auto X = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, input_size});
        auto W = make_shared<opset4::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
        auto R = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
        auto H_t = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, input_size});
        auto C_t = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
        auto lstm = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
        EXPECT_EQ(check_dynamic_lstm(lstm), true);
    }

    {
        //  batch from x, hidden from h_t
        auto X = make_shared<opset4::Parameter>(element::f32, PartialShape{batch_size, input_size});
        auto W = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
        auto R = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
        auto H_t = make_shared<opset4::Parameter>(element::f32, PartialShape{-1, hidden_size});
        auto C_t = make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
        auto lstm = make_shared<opset4::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
        EXPECT_EQ(check_dynamic_lstm(lstm), true);
    }
}
