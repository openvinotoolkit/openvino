// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

class LSTMCellV4StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v4::LSTMCell> {};

TEST_F(LSTMCellV4StaticShapeInferenceTest, default_ctor) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    const auto lstm_cell = make_op();

    input_shapes = {StaticShape{batch_size, input_size},
                    StaticShape{batch_size, hidden_size},
                    StaticShape{batch_size, hidden_size},
                    StaticShape{gates_count * hidden_size, input_size},
                    StaticShape{gates_count * hidden_size, hidden_size},
                    StaticShape{gates_count * hidden_size}},
    output_shapes = shape_inference(lstm_cell.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({batch_size, hidden_size}));
    EXPECT_EQ(output_shapes[1], StaticShape({batch_size, hidden_size}));
}

TEST_F(LSTMCellV4StaticShapeInferenceTest, basic_shape_infer) {
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
    const auto lstm_cell = make_op(X, H_t, C_t, W, R, Bias, hidden_size);

    input_shapes = {StaticShape{batch_size, input_size},
                    StaticShape{batch_size, hidden_size},
                    StaticShape{batch_size, hidden_size},
                    StaticShape{gates_count * hidden_size, input_size},
                    StaticShape{gates_count * hidden_size, hidden_size},
                    StaticShape{gates_count * hidden_size}},
    output_shapes = shape_inference(lstm_cell.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({batch_size, hidden_size}));
    EXPECT_EQ(output_shapes[1], StaticShape({batch_size, hidden_size}));
}

TEST(StaticShapeInferenceTest, LSTMCellV0Test) {
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
    const auto Peelhole = op::v0::Constant::create(element::f32, ov::Shape{3 * hidden_size}, std::vector<float>{0.f});
    const auto lstm_cell = std::make_shared<op::v0::LSTMCell>(X, H_t, C_t, W, R, Bias, Peelhole, hidden_size);

    std::vector<StaticShape> static_input_shapes = {StaticShape{batch_size, input_size},
                                                    StaticShape{batch_size, hidden_size},
                                                    StaticShape{batch_size, hidden_size},
                                                    StaticShape{gates_count * hidden_size, input_size},
                                                    StaticShape{gates_count * hidden_size, hidden_size},
                                                    StaticShape{gates_count * hidden_size},
                                                    StaticShape{3 * hidden_size}};
    const auto static_output_shapes = shape_inference(lstm_cell.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({batch_size, hidden_size}));
    ASSERT_EQ(static_output_shapes[1], StaticShape({batch_size, hidden_size}));
}
