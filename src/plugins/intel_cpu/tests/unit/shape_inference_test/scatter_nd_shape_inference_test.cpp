// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

class ScatterNDUpdateV3StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v3::ScatterNDUpdate> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(ScatterNDUpdateV3StaticShapeInferenceTest, default_ctor) {
    const auto op = make_op();

    input_shapes = StaticShapeVector{{1000, 256, 10, 13}, {25, 125, 3}, {25, 125, 13}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({1000, 256, 10, 13}));
}

TEST_F(ScatterNDUpdateV3StaticShapeInferenceTest, correct_inputs) {
    const auto d = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1, -1});
    const auto i = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1});
    const auto u = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1});

    const auto op = make_op(d, i, u);

    input_shapes = StaticShapeVector{{1000, 256, 10, 15}, {25, 125, 3}, {25, 125, 15}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({1000, 256, 10, 15}));
}

TEST_F(ScatterNDUpdateV3StaticShapeInferenceTest, params_are_dynamic_rank) {
    const auto d = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto i = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto u = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());

    const auto op = make_op(d, i, u);

    input_shapes = StaticShapeVector{{5000, 256, 10, 15}, {30, 25, 3}, {30, 25, 15}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({5000, 256, 10, 15}));
}
