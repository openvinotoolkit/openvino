// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

class LSTMSequenceV5StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v5::LSTMSequence> {};

TEST_F(LSTMSequenceV5StaticShapeInferenceTest, default_ctor) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t seq_len = 4;
    constexpr size_t num_directions = 1;
    constexpr size_t gates_count = 4;

    const auto op = std::make_shared<op::v5::LSTMSequence>();

    input_shapes = {StaticShape{batch_size, seq_len, input_size},                         // X
                    StaticShape{batch_size, num_directions, hidden_size},                 // H_t
                    StaticShape{batch_size, num_directions, hidden_size},                 // C_t
                    StaticShape{batch_size},                                              // seq_lengths
                    StaticShape{num_directions, gates_count * hidden_size, input_size},   // W
                    StaticShape{num_directions, gates_count * hidden_size, hidden_size},  // R
                    StaticShape{num_directions, gates_count * hidden_size}};              // B

    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 3);
    EXPECT_EQ(output_shapes[0], StaticShape({batch_size, num_directions, seq_len, hidden_size}));
    EXPECT_EQ(output_shapes[1], StaticShape({batch_size, num_directions, hidden_size}));
    EXPECT_EQ(output_shapes[2], StaticShape({batch_size, num_directions, hidden_size}));
}

TEST_F(LSTMSequenceV5StaticShapeInferenceTest, FORWARD) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t seq_len = 4;
    constexpr size_t num_directions = 1;
    constexpr size_t gates_count = 4;

    constexpr auto direction = op::RecurrentSequenceDirection::FORWARD;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto C_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto seq_lengths = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));

    const auto op = std::make_shared<op::v5::LSTMSequence>(X, H_t, C_t, seq_lengths, W, R, B, hidden_size, direction);

    input_shapes = {StaticShape{batch_size, seq_len, input_size},                         // X
                    StaticShape{batch_size, num_directions, hidden_size},                 // H_t
                    StaticShape{batch_size, num_directions, hidden_size},                 // C_t
                    StaticShape{batch_size},                                              // seq_lengths
                    StaticShape{num_directions, gates_count * hidden_size, input_size},   // W
                    StaticShape{num_directions, gates_count * hidden_size, hidden_size},  // R
                    StaticShape{num_directions, gates_count * hidden_size}};              // B

    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 3);
    EXPECT_EQ(output_shapes[0], StaticShape({batch_size, num_directions, seq_len, hidden_size}));
    EXPECT_EQ(output_shapes[1], StaticShape({batch_size, num_directions, hidden_size}));
    EXPECT_EQ(output_shapes[2], StaticShape({batch_size, num_directions, hidden_size}));
}

TEST_F(LSTMSequenceV5StaticShapeInferenceTest, REVERSE) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t seq_len = 4;
    constexpr size_t num_directions = 1;
    constexpr size_t gates_count = 4;

    constexpr auto direction = op::RecurrentSequenceDirection::REVERSE;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto C_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto seq_lengths = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));

    const auto op = std::make_shared<op::v5::LSTMSequence>(X, H_t, C_t, seq_lengths, W, R, B, hidden_size, direction);

    input_shapes = {StaticShape{batch_size, seq_len, input_size},                         // X
                    StaticShape{batch_size, num_directions, hidden_size},                 // H_t
                    StaticShape{batch_size, num_directions, hidden_size},                 // C_t
                    StaticShape{batch_size},                                              // seq_lengths
                    StaticShape{num_directions, gates_count * hidden_size, input_size},   // W
                    StaticShape{num_directions, gates_count * hidden_size, hidden_size},  // R
                    StaticShape{num_directions, gates_count * hidden_size}};              // B

    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes.size(), 3);
    EXPECT_EQ(output_shapes[0], StaticShape({batch_size, num_directions, seq_len, hidden_size}));
    EXPECT_EQ(output_shapes[1], StaticShape({batch_size, num_directions, hidden_size}));
    EXPECT_EQ(output_shapes[2], StaticShape({batch_size, num_directions, hidden_size}));
}

TEST_F(LSTMSequenceV5StaticShapeInferenceTest, BIDIRECTIONAL) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t seq_len = 4;
    constexpr size_t num_directions = 2;
    constexpr size_t gates_count = 4;

    constexpr auto direction = op::RecurrentSequenceDirection::BIDIRECTIONAL;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto C_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto seq_lengths = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));

    const auto op = std::make_shared<op::v5::LSTMSequence>(X, H_t, C_t, seq_lengths, W, R, B, hidden_size, direction);

    input_shapes = {StaticShape{batch_size, seq_len, input_size},                         // X
                    StaticShape{batch_size, num_directions, hidden_size},                 // H_t
                    StaticShape{batch_size, num_directions, hidden_size},                 // C_t
                    StaticShape{batch_size},                                              // seq_lengths
                    StaticShape{num_directions, gates_count * hidden_size, input_size},   // W
                    StaticShape{num_directions, gates_count * hidden_size, hidden_size},  // R
                    StaticShape{num_directions, gates_count * hidden_size}};              // B

    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes.size(), 3);
    EXPECT_EQ(output_shapes[0], StaticShape({batch_size, num_directions, seq_len, hidden_size}));
    EXPECT_EQ(output_shapes[1], StaticShape({batch_size, num_directions, hidden_size}));
    EXPECT_EQ(output_shapes[2], StaticShape({batch_size, num_directions, hidden_size}));
}
