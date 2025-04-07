// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"
#include "openvino/opsets/opset5.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, rnn_sequence_forward) {
    const size_t batch_size = 8;
    const size_t num_directions = 1;
    const size_t seq_length = 6;
    const size_t input_size = 4;
    const size_t hidden_size = 128;

    const auto X = make_shared<opset5::Parameter>(element::f32, Shape{batch_size, seq_length, input_size});
    const auto initial_hidden_state =
        make_shared<opset5::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<ov::op::v0::Parameter>(element::i32, Shape{batch_size});

    const auto W = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, hidden_size, input_size});
    const auto R = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, hidden_size, hidden_size});
    const auto B = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, hidden_size});

    const auto direction = op::RecurrentSequenceDirection::FORWARD;

    const auto sequence =
        make_shared<opset5::RNNSequence>(X, initial_hidden_state, sequence_lengths, W, R, B, hidden_size, direction);

    EXPECT_EQ(sequence->get_hidden_size(), hidden_size);
    EXPECT_EQ(sequence->get_direction(), op::RecurrentSequenceDirection::FORWARD);
    EXPECT_TRUE(sequence->get_activations_alpha().empty());
    EXPECT_TRUE(sequence->get_activations_beta().empty());
    EXPECT_EQ(sequence->get_activations()[0], "tanh");
    EXPECT_EQ(sequence->get_clip(), 0.f);
    EXPECT_EQ(sequence->get_output_element_type(0), element::f32);
    EXPECT_EQ(sequence->outputs().size(), 2);
    EXPECT_EQ(sequence->get_output_shape(0), (Shape{batch_size, num_directions, seq_length, hidden_size}));
    EXPECT_EQ(sequence->get_output_element_type(1), element::f32);
    EXPECT_EQ(sequence->get_output_shape(1), (Shape{batch_size, num_directions, hidden_size}));
}

TEST(type_prop, rnn_sequence_invalid_input) {
    const size_t batch_size = 8;
    const size_t num_directions = 1;
    const size_t seq_length = 6;
    const size_t input_size = 4;
    const size_t hidden_size = 128;

    auto X = make_shared<opset5::Parameter>(element::f32, Shape{batch_size, seq_length, input_size});
    auto H_t = make_shared<opset5::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<ov::op::v0::Parameter>(element::i32, Shape{batch_size});

    auto W = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, hidden_size, input_size});
    auto R = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, hidden_size, hidden_size});
    auto B = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, hidden_size});

    auto direction = op::RecurrentSequenceDirection::FORWARD;

    // Invalid W tensor shape.
    W = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, 2 * hidden_size, input_size});
    try {
        const auto rnn_sequence =
            make_shared<opset5::RNNSequence>(X, H_t, sequence_lengths, W, R, B, hidden_size, direction);
        FAIL() << "RNNSequence node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Second dimension of W input shape is required to be compatible with 128. Got shape: 256"));
    }

    // Invalid R tensor shape.
    W = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, hidden_size, input_size});
    R = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, hidden_size, 1});
    try {
        const auto rnn_sequence =
            make_shared<opset5::RNNSequence>(X, H_t, sequence_lengths, W, R, B, hidden_size, direction);
        FAIL() << "RNNSequence node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Dimension `hidden_size` is not matched between inputs"));
    }

    // Invalid H_t tensor shape.
    R = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, hidden_size, hidden_size});
    H_t = make_shared<opset5::Parameter>(element::f32, Shape{4, num_directions, hidden_size});
    try {
        const auto rnn_sequence =
            make_shared<opset5::RNNSequence>(X, H_t, sequence_lengths, W, R, B, hidden_size, direction);
        FAIL() << "RNNSequence node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Dimension `batch_size` is not matched between inputs"));
    }

    // Invalid B tensor shape.
    H_t = make_shared<opset5::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    B = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, 2 * hidden_size});
    try {
        const auto rnn_sequence =
            make_shared<opset5::RNNSequence>(X, H_t, sequence_lengths, W, R, B, hidden_size, direction);
        FAIL() << "RNNSequence node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Second dimension of B input shape is required to be compatible with 128. Got shape: 256"));
    }

    // Invalid direction.
    B = make_shared<opset5::Parameter>(element::f32, Shape{2, hidden_size});
    try {
        const auto rnn_sequence =
            make_shared<opset5::RNNSequence>(X, H_t, sequence_lengths, W, R, B, hidden_size, direction);
        FAIL() << "RNNSequence node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Dimension `num_directions` doesn't match to other inputs or `direction` attribute"));
    }

    // Invalid direction.
    B = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, hidden_size});
    direction = op::RecurrentSequenceDirection::BIDIRECTIONAL;
    try {
        const auto rnn_sequence =
            make_shared<opset5::RNNSequence>(X, H_t, sequence_lengths, W, R, B, hidden_size, direction);
        FAIL() << "RNNSequence node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Dimension `num_directions` doesn't match to other inputs or `direction` attribute"));
    }
}

TEST(type_prop, rnn_sequence_dynamic_inputs) {
    const auto batch_size = Dimension::dynamic();
    const size_t num_directions = 1;
    const size_t seq_length = 6;
    const auto input_size = Dimension::dynamic();
    const auto hidden_size = Dimension::dynamic();

    const auto X = make_shared<opset5::Parameter>(element::f32, PartialShape{batch_size, seq_length, input_size});
    const auto H_t =
        make_shared<opset5::Parameter>(element::f32, PartialShape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{batch_size});

    const auto W = make_shared<opset5::Parameter>(element::f32, PartialShape{num_directions, hidden_size, input_size});
    const auto R = make_shared<opset5::Parameter>(element::f32, PartialShape{num_directions, hidden_size, hidden_size});
    const auto B = make_shared<opset5::Parameter>(element::f32, PartialShape{num_directions, hidden_size});

    const auto direction = op::RecurrentSequenceDirection::REVERSE;

    const auto sequence = make_shared<opset5::RNNSequence>(X, H_t, sequence_lengths, W, R, B, 128, direction);

    EXPECT_EQ(sequence->outputs().size(), 2);
    EXPECT_EQ(sequence->get_output_element_type(0), element::f32);
    EXPECT_EQ(sequence->get_output_partial_shape(0),
              (PartialShape{batch_size, num_directions, seq_length, hidden_size}));
    EXPECT_EQ(sequence->get_output_element_type(1), element::f32);
    EXPECT_EQ(sequence->get_output_partial_shape(1), (PartialShape{batch_size, num_directions, hidden_size}));
}

TEST(type_prop, rnn_sequence_dynamic_batch_size) {
    const auto batch_size = Dimension::dynamic();
    const size_t num_directions = 1;
    const size_t seq_length = 6;
    const size_t input_size = 4;
    const size_t hidden_size = 128;

    const auto X = make_shared<opset5::Parameter>(element::f32, PartialShape{batch_size, seq_length, input_size});
    const auto H_t =
        make_shared<opset5::Parameter>(element::f32, PartialShape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{batch_size});

    const auto W = make_shared<opset5::Parameter>(element::f32, PartialShape{num_directions, hidden_size, input_size});
    const auto R = make_shared<opset5::Parameter>(element::f32, PartialShape{num_directions, hidden_size, hidden_size});
    const auto B = make_shared<opset5::Parameter>(element::f32, PartialShape{num_directions, hidden_size});

    const auto direction = op::RecurrentSequenceDirection::FORWARD;

    const auto sequence = make_shared<opset5::RNNSequence>(X, H_t, sequence_lengths, W, R, B, hidden_size, direction);

    EXPECT_EQ(sequence->outputs().size(), 2);
    EXPECT_EQ(sequence->get_output_element_type(0), element::f32);
    EXPECT_EQ(sequence->get_output_partial_shape(0),
              (PartialShape{batch_size, num_directions, seq_length, hidden_size}));
    EXPECT_EQ(sequence->get_output_element_type(1), element::f32);
    EXPECT_EQ(sequence->get_output_partial_shape(1), (PartialShape{batch_size, num_directions, hidden_size}));
}

TEST(type_prop, rnn_sequence_dynamic_input_size) {
    const size_t batch_size = 8;
    const size_t num_directions = 1;
    const size_t seq_length = 6;
    const auto input_size = Dimension::dynamic();
    const size_t hidden_size = 128;

    const auto X = make_shared<opset5::Parameter>(element::f32, PartialShape{batch_size, seq_length, input_size});
    const auto H_t =
        make_shared<opset5::Parameter>(element::f32, PartialShape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{batch_size});

    const auto W = make_shared<opset5::Parameter>(element::f32, PartialShape{num_directions, hidden_size, input_size});
    const auto R = make_shared<opset5::Parameter>(element::f32, PartialShape{num_directions, hidden_size, hidden_size});
    const auto B = make_shared<opset5::Parameter>(element::f32, PartialShape{num_directions, hidden_size});

    const auto direction = op::RecurrentSequenceDirection::FORWARD;

    const auto sequence = make_shared<opset5::RNNSequence>(X, H_t, sequence_lengths, W, R, B, hidden_size, direction);

    EXPECT_EQ(sequence->outputs().size(), 2);
    EXPECT_EQ(sequence->get_output_element_type(0), element::f32);
    EXPECT_EQ(sequence->get_output_partial_shape(0),
              (PartialShape{batch_size, num_directions, seq_length, hidden_size}));
    EXPECT_EQ(sequence->get_output_element_type(1), element::f32);
    EXPECT_EQ(sequence->get_output_partial_shape(1), (PartialShape{batch_size, num_directions, hidden_size}));
}

TEST(type_prop, rnn_sequence_dynamic_hidden_size) {
    const size_t batch_size = 8;
    const size_t num_directions = 1;
    const size_t seq_length = 6;
    const size_t input_size = 4;
    const auto hidden_size = Dimension::dynamic();

    const auto X = make_shared<opset5::Parameter>(element::f32, PartialShape{batch_size, seq_length, input_size});
    const auto H_t =
        make_shared<opset5::Parameter>(element::f32, PartialShape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{batch_size});

    const auto W = make_shared<opset5::Parameter>(element::f32, PartialShape{num_directions, hidden_size, input_size});
    const auto R = make_shared<opset5::Parameter>(element::f32, PartialShape{num_directions, hidden_size, hidden_size});
    const auto B = make_shared<opset5::Parameter>(element::f32, PartialShape{num_directions, hidden_size});

    const auto direction = op::RecurrentSequenceDirection::FORWARD;

    const auto sequence = make_shared<opset5::RNNSequence>(X, H_t, sequence_lengths, W, R, B, 128, direction);

    EXPECT_EQ(sequence->outputs().size(), 2);
    EXPECT_EQ(sequence->get_output_element_type(0), element::f32);
    EXPECT_EQ(sequence->get_output_partial_shape(0),
              (PartialShape{batch_size, num_directions, seq_length, hidden_size}));
    EXPECT_EQ(sequence->get_output_element_type(1), element::f32);
    EXPECT_EQ(sequence->get_output_partial_shape(1), (PartialShape{batch_size, num_directions, hidden_size}));
}

TEST(type_prop, rnn_sequence_dynamic_invalid_input_rank0) {
    const size_t batch_size = 8;
    const size_t num_directions = 1;
    const size_t seq_length = 6;
    const size_t input_size = 4;
    const size_t hidden_size = 128;

    auto X = make_shared<opset5::Parameter>(element::f32, Shape{batch_size, seq_length, input_size});
    auto H_t = make_shared<opset5::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<ov::op::v0::Parameter>(element::i32, Shape{batch_size});

    auto W = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, hidden_size, input_size});
    auto R = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, hidden_size, hidden_size});
    auto B = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, hidden_size});

    const auto direction = op::RecurrentSequenceDirection::FORWARD;

    // Invalid rank0 for X tensor.
    X = make_shared<opset5::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(
        const auto unused = make_shared<opset5::RNNSequence>(X, H_t, sequence_lengths, W, R, B, hidden_size, direction),
        ov::AssertFailure)
        << "RNNSequence node was created with invalid data.";

    // Invalid rank0 for H_t tensor.
    X = make_shared<opset5::Parameter>(element::f32, Shape{batch_size, seq_length, input_size});
    H_t = make_shared<opset5::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(
        const auto unused = make_shared<opset5::RNNSequence>(X, H_t, sequence_lengths, W, R, B, hidden_size, direction),
        ov::AssertFailure)
        << "RNNSequence node was created with invalid data.";

    // Invalid rank0 for W tensor.
    H_t = make_shared<opset5::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    W = make_shared<opset5::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(
        const auto unused = make_shared<opset5::RNNSequence>(X, H_t, sequence_lengths, W, R, B, hidden_size, direction),
        ov::AssertFailure)
        << "RNNSequence node was created with invalid data.";

    // Invalid rank0 for R tensor.
    W = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, hidden_size, input_size});
    R = make_shared<opset5::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(
        const auto unused = make_shared<opset5::RNNSequence>(X, H_t, sequence_lengths, W, R, B, hidden_size, direction),
        ov::AssertFailure)
        << "RNNSequence node was created with invalid data.";

    // Invalid rank0 for B tensor.
    R = make_shared<opset5::Parameter>(element::f32, Shape{num_directions, hidden_size, hidden_size});
    B = make_shared<opset5::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(
        const auto unused = make_shared<opset5::RNNSequence>(X, H_t, sequence_lengths, W, R, B, hidden_size, direction),
        ov::AssertFailure)
        << "RNNSequence node was created with invalid data.";
}

TEST(type_prop, rnn_sequence_input_dynamic_rank) {
    const int64_t batch_size = 8;
    const int64_t num_directions = 1;
    const int64_t seq_length = 6;
    const int64_t input_size = 4;
    const int64_t hidden_size = 128;

    auto X = make_shared<opset5::Parameter>(element::f32, PartialShape{batch_size, seq_length, input_size});
    auto H_t = make_shared<opset5::Parameter>(element::f32, PartialShape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{batch_size});

    auto W = make_shared<opset5::Parameter>(element::f32, PartialShape{num_directions, hidden_size, input_size});
    auto R = make_shared<opset5::Parameter>(element::f32, PartialShape{num_directions, hidden_size, hidden_size});
    auto B = make_shared<opset5::Parameter>(element::f32, PartialShape{num_directions, hidden_size});

    const auto direction = op::RecurrentSequenceDirection::FORWARD;

    auto check_dynamic_rnn = [=](const shared_ptr<opset5::RNNSequence>& rnn) -> bool {
        return rnn->output(0).get_partial_shape() ==
                   PartialShape{batch_size, num_directions, seq_length, hidden_size} &&
               rnn->output(0).get_element_type() == rnn->input(0).get_element_type() &&
               rnn->output(1).get_partial_shape() == PartialShape{batch_size, num_directions, hidden_size} &&
               rnn->output(1).get_element_type() == rnn->input(0).get_element_type();
    };

    X = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto rnn_x = make_shared<opset5::RNNSequence>(X,
                                                  H_t,
                                                  sequence_lengths,
                                                  W,
                                                  R,
                                                  B,
                                                  static_cast<size_t>(hidden_size),
                                                  direction);
    EXPECT_EQ(rnn_x->get_output_partial_shape(0), (PartialShape{batch_size, num_directions, -1, hidden_size}));
    EXPECT_EQ(rnn_x->get_output_partial_shape(1), (PartialShape{batch_size, num_directions, hidden_size}));

    X = make_shared<opset5::Parameter>(element::f32, PartialShape{batch_size, seq_length, input_size});
    H_t = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto rnn_h = make_shared<opset5::RNNSequence>(X,
                                                  H_t,
                                                  sequence_lengths,
                                                  W,
                                                  R,
                                                  B,
                                                  static_cast<size_t>(hidden_size),
                                                  direction);
    EXPECT_EQ(check_dynamic_rnn(rnn_h), true);

    H_t = make_shared<opset5::Parameter>(element::f32, PartialShape{batch_size, num_directions, hidden_size});
    W = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto rnn_w = make_shared<opset5::RNNSequence>(X,
                                                  H_t,
                                                  sequence_lengths,
                                                  W,
                                                  R,
                                                  B,
                                                  static_cast<size_t>(hidden_size),
                                                  direction);
    EXPECT_EQ(check_dynamic_rnn(rnn_w), true);

    W = make_shared<opset5::Parameter>(element::f32, PartialShape{num_directions, hidden_size, input_size});
    R = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto rnn_r = make_shared<opset5::RNNSequence>(X,
                                                  H_t,
                                                  sequence_lengths,
                                                  W,
                                                  R,
                                                  B,
                                                  static_cast<size_t>(hidden_size),
                                                  direction);
    EXPECT_EQ(check_dynamic_rnn(rnn_r), true);

    R = make_shared<opset5::Parameter>(element::f32, PartialShape{num_directions, hidden_size, hidden_size});
    B = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto rnn_b = make_shared<opset5::RNNSequence>(X,
                                                  H_t,
                                                  sequence_lengths,
                                                  W,
                                                  R,
                                                  B,
                                                  static_cast<size_t>(hidden_size),
                                                  direction);
    EXPECT_EQ(check_dynamic_rnn(rnn_b), true);
}
