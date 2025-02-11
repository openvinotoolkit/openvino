// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

class GRUSequenceV5StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v5::GRUSequence> {};

TEST_F(GRUSequenceV5StaticShapeInferenceTest, default_ctor) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t seq_len = 4;
    constexpr size_t num_directions = 1;
    constexpr size_t gates_count = 3;

    const auto gru_sequence = make_op();

    input_shapes = {StaticShape{batch_size, seq_len, input_size},                         // X
                    StaticShape{batch_size, num_directions, hidden_size},                 // H_t
                    StaticShape{batch_size},                                              // seq_lengths
                    StaticShape{num_directions, gates_count * hidden_size, input_size},   // W
                    StaticShape{num_directions, gates_count * hidden_size, hidden_size},  // R
                    StaticShape{num_directions, gates_count * hidden_size}};              // B

    output_shapes = shape_inference(gru_sequence.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({batch_size, num_directions, seq_len, hidden_size}));
    EXPECT_EQ(output_shapes[1], StaticShape({batch_size, num_directions, hidden_size}));
}

TEST_F(GRUSequenceV5StaticShapeInferenceTest, FORWARD) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t seq_len = 4;
    constexpr size_t num_directions = 1;
    constexpr size_t gates_count = 3;

    constexpr auto direction = op::RecurrentSequenceDirection::FORWARD;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto seq_lengths = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));

    const auto gru_sequence =
        std::make_shared<op::v5::GRUSequence>(X, H_t, seq_lengths, W, R, B, hidden_size, direction);

    input_shapes = {StaticShape{batch_size, seq_len, input_size},                         // X
                    StaticShape{batch_size, num_directions, hidden_size},                 // H_t
                    StaticShape{batch_size},                                              // seq_lengths
                    StaticShape{num_directions, gates_count * hidden_size, input_size},   // W
                    StaticShape{num_directions, gates_count * hidden_size, hidden_size},  // R
                    StaticShape{num_directions, gates_count * hidden_size}};              // B

    output_shapes = shape_inference(gru_sequence.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({batch_size, num_directions, seq_len, hidden_size}));
    EXPECT_EQ(output_shapes[1], StaticShape({batch_size, num_directions, hidden_size}));
}

TEST_F(GRUSequenceV5StaticShapeInferenceTest, FORWARD_linear_before) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t seq_len = 4;
    constexpr size_t num_directions = 1;
    constexpr size_t gates_count = 3;

    constexpr auto direction = op::RecurrentSequenceDirection::FORWARD;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto seq_lengths = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));

    const auto gru_sequence = std::make_shared<op::v5::GRUSequence>(X,
                                                                    H_t,
                                                                    seq_lengths,
                                                                    W,
                                                                    R,
                                                                    B,
                                                                    hidden_size,
                                                                    direction,
                                                                    std::vector<std::string>{"sigmoid", "tanh"},
                                                                    std::vector<float>{},
                                                                    std::vector<float>{},
                                                                    0.f,
                                                                    true);

    input_shapes = {StaticShape{batch_size, seq_len, input_size},                         // X
                    StaticShape{batch_size, num_directions, hidden_size},                 // H_t
                    StaticShape{batch_size},                                              // seq_lengths
                    StaticShape{num_directions, gates_count * hidden_size, input_size},   // W
                    StaticShape{num_directions, gates_count * hidden_size, hidden_size},  // R
                    StaticShape{num_directions, (gates_count + 1) * hidden_size}};        // B

    output_shapes = shape_inference(gru_sequence.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({batch_size, num_directions, seq_len, hidden_size}));
    EXPECT_EQ(output_shapes[1], StaticShape({batch_size, num_directions, hidden_size}));
}

TEST_F(GRUSequenceV5StaticShapeInferenceTest, REVERSE) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t seq_len = 4;
    constexpr size_t num_directions = 1;
    constexpr size_t gates_count = 3;

    constexpr auto direction = op::RecurrentSequenceDirection::REVERSE;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto seq_lengths = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));

    const auto gru_sequence =
        std::make_shared<op::v5::GRUSequence>(X, H_t, seq_lengths, W, R, B, hidden_size, direction);

    input_shapes = {StaticShape{batch_size, seq_len, input_size},                         // X
                    StaticShape{batch_size, num_directions, hidden_size},                 // H_t
                    StaticShape{batch_size},                                              // seq_lengths
                    StaticShape{num_directions, gates_count * hidden_size, input_size},   // W
                    StaticShape{num_directions, gates_count * hidden_size, hidden_size},  // R
                    StaticShape{num_directions, gates_count * hidden_size}};              // B

    output_shapes = shape_inference(gru_sequence.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({batch_size, num_directions, seq_len, hidden_size}));
    EXPECT_EQ(output_shapes[1], StaticShape({batch_size, num_directions, hidden_size}));
}

TEST_F(GRUSequenceV5StaticShapeInferenceTest, BIDIRECTIONAL) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t seq_len = 4;
    constexpr size_t num_directions = 2;
    constexpr size_t gates_count = 3;

    constexpr auto direction = op::RecurrentSequenceDirection::BIDIRECTIONAL;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto seq_lengths = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));

    const auto gru_sequence =
        std::make_shared<op::v5::GRUSequence>(X, H_t, seq_lengths, W, R, B, hidden_size, direction);

    input_shapes = {StaticShape{batch_size, seq_len, input_size},                         // X
                    StaticShape{batch_size, num_directions, hidden_size},                 // H_t
                    StaticShape{batch_size},                                              // seq_lengths
                    StaticShape{num_directions, gates_count * hidden_size, input_size},   // W
                    StaticShape{num_directions, gates_count * hidden_size, hidden_size},  // R
                    StaticShape{num_directions, gates_count * hidden_size}};              // B

    output_shapes = shape_inference(gru_sequence.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({batch_size, num_directions, seq_len, hidden_size}));
    EXPECT_EQ(output_shapes[1], StaticShape({batch_size, num_directions, hidden_size}));
}
