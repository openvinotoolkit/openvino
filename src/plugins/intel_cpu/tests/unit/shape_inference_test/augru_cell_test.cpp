// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "ov_ops/augru_cell.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, AUGRUCellTest_all_inputs_static_rank) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t gates_count = 3;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    const auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));

    const auto augru = std::make_shared<ov::op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);

    std::vector<StaticShape> static_input_shapes{StaticShape{batch_size, input_size},                  // X
                                                 StaticShape{batch_size, hidden_size},                 // H_t
                                                 StaticShape{gates_count * hidden_size, input_size},   // W
                                                 StaticShape{gates_count * hidden_size, hidden_size},  // R
                                                 StaticShape{gates_count * hidden_size},               // B
                                                 StaticShape{batch_size, 1}};                          // A

    std::vector<StaticShape> static_output_shapes{StaticShape{}, StaticShape{}};

    static_output_shapes = shape_inference(augru.get(), static_input_shapes);
    EXPECT_EQ(static_output_shapes[0], StaticShape({batch_size, hidden_size}));
}

TEST(StaticShapeInferenceTest, AUGRUCellTest_all_inputs_dynamic_rank) {
    constexpr size_t batch_size = 2;
    constexpr size_t input_size = 3;
    constexpr size_t hidden_size = 5;
    constexpr size_t gates_count = 3;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());

    const auto augru = std::make_shared<ov::op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);

    std::vector<StaticShape> static_input_shapes{StaticShape{batch_size, input_size},                  // X
                                                 StaticShape{batch_size, hidden_size},                 // H_t
                                                 StaticShape{gates_count * hidden_size, input_size},   // W
                                                 StaticShape{gates_count * hidden_size, hidden_size},  // R
                                                 StaticShape{gates_count * hidden_size},               // B
                                                 StaticShape{batch_size, 1}};                          // A

    std::vector<StaticShape> static_output_shapes{StaticShape{}, StaticShape{}};

    static_output_shapes = shape_inference(augru.get(), static_input_shapes);
    EXPECT_EQ(static_output_shapes[0], StaticShape({batch_size, hidden_size}));
}
