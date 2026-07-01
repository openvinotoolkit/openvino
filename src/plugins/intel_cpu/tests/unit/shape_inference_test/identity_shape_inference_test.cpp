// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "identity_shape_inference.hpp"

#include <gtest/gtest.h>

#include "utils.hpp"

using namespace ov::intel_cpu;

class Identityv16StaticShapeInferenceTest : public OpStaticShapeInferenceTest<ov::op::v16::Identity> {
protected:
};

TEST_F(Identityv16StaticShapeInferenceTest, Identity_default_ctor) {
    op = make_op();
    input_shapes = StaticShapeVector{{2, 2}};

    auto output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({2, 2}));
}

TEST_F(Identityv16StaticShapeInferenceTest, Identity_4_4_small_matrix) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(2));
    auto op = make_op(data);
    input_shapes = StaticShapeVector{{4, 4}};

    auto output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({4, 4}));
}

TEST_F(Identityv16StaticShapeInferenceTest, Identity_10_10_big_matrix) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(2));
    auto op = make_op(data);
    input_shapes = StaticShapeVector{{10, 10}};

    auto output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({10, 10}));
}

TEST_F(Identityv16StaticShapeInferenceTest, Identity_10_1_1_keep_batch_when_single_cell_matrix) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));
    auto op = make_op(data);
    input_shapes = StaticShapeVector{{10, 1, 1}};

    auto output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({10, 1, 1}));
}

TEST_F(Identityv16StaticShapeInferenceTest, Identity_10_9_9_keep_batch_big_matrix) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));
    auto op = make_op(data);
    input_shapes = StaticShapeVector{{10, 9, 9}};

    auto output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({10, 9, 9}));
}

TEST_F(Identityv16StaticShapeInferenceTest, Identity_10_5_3_2_2_complex_multi_dim_matrix) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(5));
    auto op = make_op(data);
    input_shapes = StaticShapeVector{{10, 5, 3, 2, 2}};

    auto output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({10, 5, 3, 2, 2}));
}
