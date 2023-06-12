// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;


TEST(StaticShapeInferenceTest, RNNCellTest_default_ctor) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t gates_count = 1;

    // Default `B` input is created as Constant by RNNCell contructor
    const auto gru = std::make_shared<op::v0::RNNCell>();

    std::vector<StaticShape> static_input_shapes{StaticShape{batch_size, input_size},                  // X
                                                 StaticShape{batch_size, hidden_size},                 // H_t
                                                 StaticShape{gates_count * hidden_size, input_size},   // W
                                                 StaticShape{gates_count * hidden_size, hidden_size},  // R
                                                 StaticShape{gates_count * hidden_size}};              // B

    std::vector<StaticShape> static_output_shapes;
    shape_inference(gru.get(), static_input_shapes, static_output_shapes);
    EXPECT_EQ(static_output_shapes[0], StaticShape({batch_size, hidden_size}));
}

TEST(StaticShapeInferenceTest, RNNCellTest_default_bias) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t gates_count = 1;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));

    // Default `B` input is created as Constant by RNNCell contructor
    const auto gru = std::make_shared<op::v0::RNNCell>(X, H_t, W, R, hidden_size);

    std::vector<StaticShape> static_input_shapes{StaticShape{batch_size, input_size},                  // X
                                                 StaticShape{batch_size, hidden_size},                 // H_t
                                                 StaticShape{gates_count * hidden_size, input_size},   // W
                                                 StaticShape{gates_count * hidden_size, hidden_size},  // R
                                                 StaticShape{gates_count * hidden_size}};              // B

    std::vector<StaticShape> static_output_shapes;
    shape_inference(gru.get(), static_input_shapes, static_output_shapes);
    EXPECT_EQ(static_output_shapes[0], StaticShape({batch_size, hidden_size}));
}

TEST(StaticShapeInferenceTest, RNNCellTest_with_bias) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t gates_count = 1;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    const auto gru = std::make_shared<op::v0::RNNCell>(X, H_t, W, R, B, hidden_size);

    std::vector<StaticShape> static_input_shapes{StaticShape{batch_size, input_size},                  // X
                                                 StaticShape{batch_size, hidden_size},                 // H_t
                                                 StaticShape{gates_count * hidden_size, input_size},   // W
                                                 StaticShape{gates_count * hidden_size, hidden_size},  // R
                                                 StaticShape{gates_count * hidden_size}};              // B

    std::vector<StaticShape> static_output_shapes;
    shape_inference(gru.get(), static_input_shapes, static_output_shapes);
    EXPECT_EQ(static_output_shapes[0], StaticShape({batch_size, hidden_size}));
}

TEST(StaticShapeInferenceTest, RNNCellTest_dynamic_rank_inputs) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t gates_count = 1;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());

    const auto gru = std::make_shared<op::v0::RNNCell>(X, H_t, W, R, B, hidden_size);

    std::vector<StaticShape> static_input_shapes{StaticShape{batch_size, input_size},                  // X
                                                 StaticShape{batch_size, hidden_size},                 // H_t
                                                 StaticShape{gates_count * hidden_size, input_size},   // W
                                                 StaticShape{gates_count * hidden_size, hidden_size},  // R
                                                 StaticShape{gates_count * hidden_size}};              // B

    std::vector<StaticShape> static_output_shapes;
    shape_inference(gru.get(), static_input_shapes, static_output_shapes);
    EXPECT_EQ(static_output_shapes[0], StaticShape({batch_size, hidden_size}));
}
