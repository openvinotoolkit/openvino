// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, LstmCellTest) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    const auto C_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    const auto Bias = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1});
    const auto lstm_cell = std::make_shared<op::v4::LSTMCell>(X, H_t, C_t, W, R, Bias, hidden_size);

    std::vector<StaticShape> static_input_shapes = {StaticShape{batch_size, input_size},
                                                    StaticShape{batch_size, hidden_size},
                                                    StaticShape{batch_size, hidden_size},
                                                    StaticShape{gates_count * hidden_size, input_size},
                                                    StaticShape{gates_count * hidden_size, hidden_size},
                                                    StaticShape{gates_count * hidden_size}},
                             static_output_shapes = {StaticShape{}, StaticShape{}};
    shape_inference(lstm_cell.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({batch_size, hidden_size}));
    ASSERT_EQ(static_output_shapes[1], StaticShape({batch_size, hidden_size}));
}

TEST(StaticShapeInferenceTest, LstmCellV1Test) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    const auto X = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    const auto W = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    const auto R = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    const auto H_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    const auto C_t = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    const auto Bias = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1});
    const auto Peelhole = op::v0::Constant::create(element::f32, Shape{3 * hidden_size}, std::vector<float>{0.f});
    const auto lstm_cell = std::make_shared<op::v0::LSTMCell>(X, H_t, C_t, W, R, Bias, Peelhole, hidden_size);

    std::vector<StaticShape> static_input_shapes = {StaticShape{batch_size, input_size},
                                                    StaticShape{batch_size, hidden_size},
                                                    StaticShape{batch_size, hidden_size},
                                                    StaticShape{gates_count * hidden_size, input_size},
                                                    StaticShape{gates_count * hidden_size, hidden_size},
                                                    StaticShape{gates_count * hidden_size},
                                                    StaticShape{3 * hidden_size}},
                             static_output_shapes = {StaticShape{}, StaticShape{}};
    shape_inference(lstm_cell.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({batch_size, hidden_size}));
    ASSERT_EQ(static_output_shapes[1], StaticShape({batch_size, hidden_size}));
}