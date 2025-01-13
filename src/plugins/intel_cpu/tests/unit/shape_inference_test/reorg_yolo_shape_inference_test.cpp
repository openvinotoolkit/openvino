
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "reorg_yolo_shape_inference.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class StaticShapeReorgYoloTest : public OpStaticShapeInferenceTest<op::v0::ReorgYolo> {};

TEST_F(StaticShapeReorgYoloTest, default_ctor_no_args) {
    op = make_op();
    op->set_strides(3);

    input_shapes = StaticShapeVector{{2, 9, 12, 6}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 81, 4, 2}));
}

TEST_F(StaticShapeReorgYoloTest, data_input_is_dynamic_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    op = make_op(data, 2);

    input_shapes = StaticShapeVector{{2, 12, 12, 24}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 48, 6, 12}));
}

TEST_F(StaticShapeReorgYoloTest, data_input_is_static_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic(4));
    op = make_op(data, 2);

    input_shapes = StaticShapeVector{{2, 20, 12, 24}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 80, 6, 12}));
}

TEST_F(StaticShapeReorgYoloTest, data_shape_not_compatible_rank_4) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    op = make_op(data, 2);

    OV_EXPECT_THROW(shape_inference(op.get(), StaticShapeVector({{2, 20, 12, 24, 1}})),
                    NodeValidationFailure,
                    HasSubstr("[N, C, H, W] input shape is required"));
}

TEST_F(StaticShapeReorgYoloTest, h_dim_not_div_by_stride) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    op = make_op(data, 2);

    OV_EXPECT_THROW(shape_inference(op.get(), StaticShapeVector{{2, 20, 11, 24}}),
                    NodeValidationFailure,
                    HasSubstr("H and W should be divisible by stride"));
}
