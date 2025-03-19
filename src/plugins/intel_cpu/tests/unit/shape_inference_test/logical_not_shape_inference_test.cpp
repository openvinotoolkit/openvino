// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/parameter.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class LogicalNotStaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v1::LogicalNot> {};

TEST_F(LogicalNotStaticShapeInferenceTest, static_rank) {
    const auto a = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{-1, -1, -1, -1});
    const auto op = this->make_op(a);

    this->input_shapes = {StaticShape{3, 4, 7, 5}};

    this->output_shapes = shape_inference(op.get(), this->input_shapes);

    ASSERT_EQ(this->output_shapes.front(), StaticShape({3, 4, 7, 5}));
}

TEST_F(LogicalNotStaticShapeInferenceTest, dynamic_rank) {
    const auto a = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto op = this->make_op(a);

    this->input_shapes = {StaticShape{3, 1, 5, 2}};

    this->output_shapes = shape_inference(op.get(), this->input_shapes);

    ASSERT_EQ(this->output_shapes.front(), StaticShape({3, 1, 5, 2}));
}
