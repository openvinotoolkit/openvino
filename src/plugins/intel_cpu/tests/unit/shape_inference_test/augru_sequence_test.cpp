// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "ov_ops/augru_sequence.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, AGRUSequenceTest_FORWARD_all_static_rank) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t seq_len = 4;
    constexpr size_t num_directions = 1;
    constexpr size_t gates_count = 3;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto seq_lengths = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    const auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));

    const auto augru_sequence =
        std::make_shared<ov::op::internal::AUGRUSequence>(X, H_t, seq_lengths, W, R, B, A, hidden_size);

    std::vector<StaticShape> static_input_shapes{
        StaticShape{batch_size, seq_len, input_size},                         // X
        StaticShape{batch_size, num_directions, hidden_size},                 // H_t
        StaticShape{batch_size},                                              // seq_lengths
        StaticShape{num_directions, gates_count * hidden_size, input_size},   // W
        StaticShape{num_directions, gates_count * hidden_size, hidden_size},  // R
        StaticShape{num_directions, gates_count * hidden_size},               // B
        StaticShape{batch_size, seq_len, 1}};                                 // A

    const auto static_output_shapes = shape_inference(augru_sequence.get(), static_input_shapes);
    EXPECT_EQ(static_output_shapes[0], StaticShape({batch_size, num_directions, seq_len, hidden_size}));
    EXPECT_EQ(static_output_shapes[1], StaticShape({batch_size, num_directions, hidden_size}));
}

TEST(StaticShapeInferenceTest, AGRUSequenceTest_FORWARD_all_inputs_dynamic_rank) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t seq_len = 4;
    constexpr size_t num_directions = 1;
    constexpr size_t gates_count = 3;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto seq_lengths = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());

    const auto augru_sequence =
        std::make_shared<ov::op::internal::AUGRUSequence>(X, H_t, seq_lengths, W, R, B, A, hidden_size);

    std::vector<StaticShape> static_input_shapes{
        StaticShape{batch_size, seq_len, input_size},                         // X
        StaticShape{batch_size, num_directions, hidden_size},                 // H_t
        StaticShape{batch_size},                                              // seq_lengths
        StaticShape{num_directions, gates_count * hidden_size, input_size},   // W
        StaticShape{num_directions, gates_count * hidden_size, hidden_size},  // R
        StaticShape{num_directions, gates_count * hidden_size},               // B
        StaticShape{batch_size, seq_len, 1}};                                 // A

    const auto static_output_shapes = shape_inference(augru_sequence.get(), static_input_shapes);
    EXPECT_EQ(static_output_shapes[0], StaticShape({batch_size, num_directions, seq_len, hidden_size}));
    EXPECT_EQ(static_output_shapes[1], StaticShape({batch_size, num_directions, hidden_size}));
}
