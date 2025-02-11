// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "gather_elements_shape_inference.hpp"
#include "openvino/op/ops.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class GatherElementsStaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v6::GatherElements> {};

TEST_F(GatherElementsStaticShapeInferenceTest, GatherElements_basic) {
    int64_t axis = -1;
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1, -1});

    op = make_op(data, indices, axis);
    input_shapes = {StaticShape{300, 3, 10, 2}, StaticShape{300, 3, 10, 33333}};
    output_shapes = {StaticShape{}};

    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], (StaticShape{300, 3, 10, 33333}));
}

TEST_F(GatherElementsStaticShapeInferenceTest, GatherElements_incompatible_rank) {
    int64_t axis = -1;
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());

    op = make_op(data, indices, axis);
    input_shapes = {StaticShape{1, 2, 3, 4, 5}, StaticShape{1, 2, 3, 4}};
    output_shapes = {StaticShape{}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    ov::NodeValidationFailure,
                    HasSubstr("rank must be equal"));
}

TEST_F(GatherElementsStaticShapeInferenceTest, GatherElements_incompatible_dims) {
    int64_t axis = -1;
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1, -1});

    op = make_op(data, indices, axis);
    input_shapes = {StaticShape{300, 4, 10, 2}, StaticShape{300, 5, 10, 33333}};
    output_shapes = {StaticShape{}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    ov::NodeValidationFailure,
                    HasSubstr("are not consistent"));
}

TEST_F(GatherElementsStaticShapeInferenceTest, GatherElements_default_constructor) {
    int64_t axis = -1;
    op = make_op();
    op->set_axis(axis);
    input_shapes = {StaticShape{300, 3, 10, 2}, StaticShape{300, 3, 10, 33333}};
    output_shapes = {StaticShape{}};

    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], (StaticShape{300, 3, 10, 33333}));
}
