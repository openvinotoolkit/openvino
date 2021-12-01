// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <lstm_cell_shape_inference.hpp>

#include "utils/shape_inference/static_shape.hpp"

using namespace ov;

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
    std::vector<PartialShape> input_shapes = {PartialShape{batch_size, input_size},
                                              PartialShape{batch_size, hidden_size},
                                              PartialShape{batch_size, hidden_size},
                                              PartialShape{gates_count * hidden_size, input_size},
                                              PartialShape{gates_count * hidden_size, hidden_size},
                                              PartialShape{gates_count * hidden_size}},
                              output_shapes = {PartialShape{}, PartialShape{}};
    shape_infer(lstm_cell.get(), input_shapes, output_shapes);
    ASSERT_EQ(output_shapes[0], PartialShape({batch_size, hidden_size}));
    ASSERT_EQ(output_shapes[1], PartialShape({batch_size, hidden_size}));
    std::vector<StaticShape> static_input_shapes = {StaticShape{batch_size, input_size},
                                                    StaticShape{batch_size, hidden_size},
                                                    StaticShape{batch_size, hidden_size},
                                                    StaticShape{gates_count * hidden_size, input_size},
                                                    StaticShape{gates_count * hidden_size, hidden_size},
                                                    StaticShape{gates_count * hidden_size}},
                             static_output_shapes = {StaticShape{}, StaticShape{}};
    shape_infer(lstm_cell.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({batch_size, hidden_size}));
    ASSERT_EQ(static_output_shapes[1], StaticShape({batch_size, hidden_size}));
}