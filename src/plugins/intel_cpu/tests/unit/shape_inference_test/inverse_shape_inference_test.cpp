// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "inverse_shape_inference.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

class Inversev14StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v14::Inverse> {
protected:
};

TEST_F(Inversev14StaticShapeInferenceTest, inverse_default_ctor) {
    op = make_op();
    op->set_adjoint(false);

    input_shapes = StaticShapeVector{{2, 2}};
    auto output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({2, 2}));
}

TEST_F(Inversev14StaticShapeInferenceTest, inverse_4_4_small_matrix) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    auto inverse = std::make_shared<op::v14::Inverse>(data, false);

    input_shapes = StaticShapeVector{{4, 4}};
    auto output_shapes = shape_inference(inverse.get(), input_shapes);
    ASSERT_EQ(output_shapes[0], StaticShape({4, 4}));
}

TEST_F(Inversev14StaticShapeInferenceTest, inverse_10_10_big_matrix) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    auto inverse = std::make_shared<op::v14::Inverse>(data, false);

    input_shapes = StaticShapeVector{{10, 10}};
    auto output_shapes = shape_inference(inverse.get(), input_shapes);
    ASSERT_EQ(output_shapes[0], StaticShape({10, 10}));
}

TEST_F(Inversev14StaticShapeInferenceTest, inverse_10_1_1_keep_batch_when_single_cell_matrix) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    auto inverse = std::make_shared<op::v14::Inverse>(data, false);

    input_shapes = StaticShapeVector{{10, 1, 1}};
    auto output_shapes = shape_inference(inverse.get(), input_shapes);
    ASSERT_EQ(output_shapes[0], StaticShape({10, 1, 1}));
}

TEST_F(Inversev14StaticShapeInferenceTest, inverse_10_9_9_keep_batch_big_matrix) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    auto inverse = std::make_shared<op::v14::Inverse>(data, false);

    input_shapes = StaticShapeVector{{10, 9, 9}};
    auto output_shapes = shape_inference(inverse.get(), input_shapes);
    ASSERT_EQ(output_shapes[0], StaticShape({10, 9, 9}));
}

TEST_F(Inversev14StaticShapeInferenceTest, inverse_10_5_3_2_2_complex_multi_dim_matrix) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    auto inverse = std::make_shared<op::v14::Inverse>(data, false);

    input_shapes = StaticShapeVector{{10, 5, 3, 2, 2}};
    auto output_shapes = shape_inference(inverse.get(), input_shapes);
    ASSERT_EQ(output_shapes[0], StaticShape({10, 5, 3, 2, 2}));
}
