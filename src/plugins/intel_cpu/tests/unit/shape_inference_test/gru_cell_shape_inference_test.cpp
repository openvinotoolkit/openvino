// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

class GRUCellV3StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v3::GRUCell> {};

TEST_F(GRUCellV3StaticShapeInferenceTest, default_ctor) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t gates_count = 3;

    const auto gru = make_op();

    input_shapes = {StaticShape{batch_size, input_size},                  // X
                    StaticShape{batch_size, hidden_size},                 // H_t
                    StaticShape{gates_count * hidden_size, input_size},   // W
                    StaticShape{gates_count * hidden_size, hidden_size},  // R
                    StaticShape{gates_count * hidden_size}};              // B

    output_shapes = shape_inference(gru.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({batch_size, hidden_size}));
}

TEST_F(GRUCellV3StaticShapeInferenceTest, default_bias) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t gates_count = 3;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));

    // Default `B` input is created as Constant by GRUCell contructor
    const auto gru = make_op(X, H_t, W, R, hidden_size);

    input_shapes = {StaticShape{batch_size, input_size},                  // X
                    StaticShape{batch_size, hidden_size},                 // H_t
                    StaticShape{gates_count * hidden_size, input_size},   // W
                    StaticShape{gates_count * hidden_size, hidden_size},  // R
                    StaticShape{gates_count * hidden_size}};              // B

    output_shapes = shape_inference(gru.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({batch_size, hidden_size}));
}

TEST_F(GRUCellV3StaticShapeInferenceTest, with_bias) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t gates_count = 3;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    const auto gru = make_op(X, H_t, W, R, B, hidden_size);

    input_shapes = {StaticShape{batch_size, input_size},                  // X
                    StaticShape{batch_size, hidden_size},                 // H_t
                    StaticShape{gates_count * hidden_size, input_size},   // W
                    StaticShape{gates_count * hidden_size, hidden_size},  // R
                    StaticShape{gates_count * hidden_size}};              // B

    output_shapes = {StaticShape{}, StaticShape{}};

    output_shapes = shape_inference(gru.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({batch_size, hidden_size}));
}

TEST_F(GRUCellV3StaticShapeInferenceTest, linear_before) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t gates_count = 3;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));

    const auto gru = make_op(X,
                             H_t,
                             W,
                             R,
                             B,
                             hidden_size,
                             std::vector<std::string>{"sigmoid", "tanh"},
                             std::vector<float>{},
                             std::vector<float>{},
                             0.f,
                             true);

    input_shapes = {StaticShape{batch_size, input_size},                  // X
                    StaticShape{batch_size, hidden_size},                 // H_t
                    StaticShape{gates_count * hidden_size, input_size},   // W
                    StaticShape{gates_count * hidden_size, hidden_size},  // R
                    StaticShape{(gates_count + 1) * hidden_size}};        // B

    output_shapes = {StaticShape{}, StaticShape{}};

    output_shapes = shape_inference(gru.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({batch_size, hidden_size}));
}

TEST_F(GRUCellV3StaticShapeInferenceTest, dynamic_rank_inputs) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t gates_count = 3;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());

    const auto gru = make_op(X, H_t, W, R, B, hidden_size);

    input_shapes = {StaticShape{batch_size, input_size},                  // X
                    StaticShape{batch_size, hidden_size},                 // H_t
                    StaticShape{gates_count * hidden_size, input_size},   // W
                    StaticShape{gates_count * hidden_size, hidden_size},  // R
                    StaticShape{gates_count * hidden_size}};              // B

    output_shapes = shape_inference(gru.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({batch_size, hidden_size}));
}
