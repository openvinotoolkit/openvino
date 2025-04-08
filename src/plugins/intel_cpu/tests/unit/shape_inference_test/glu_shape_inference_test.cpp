// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "ov_ops/glu.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using testing::HasSubstr;

TEST(StaticShapeInferenceTest, GLUStaticShapeInferenceTestDefaultCtor) {
    constexpr int64_t axis = -1;
    constexpr int64_t split_lengths = 48;

    const auto op = std::make_shared<op::internal::GLU>();
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());

    op->set_arguments(ov::OutputVector{data});
    op->set_axis(axis);
    op->set_split_lengths(split_lengths);

    std::vector<StaticShape> static_input_shapes = {StaticShape{20, 1, 96}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes.size(), 1);
    EXPECT_EQ(static_output_shapes[0], StaticShape({20, 1, 48}));
}

TEST(StaticShapeInferenceTest, GLUStaticShapeInferenceTestBasic) {
    constexpr int64_t axis = -1;
    constexpr int64_t split_lengths = 48;
    const auto glu_type = ov::op::internal::GLU::GluType::Swish;

    const auto data = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto op = std::make_shared<op::internal::GLU>(data, axis, split_lengths, glu_type, 1);

    std::vector<StaticShape> static_input_shapes = {StaticShape{20, 1, 96}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes.size(), 1);
    EXPECT_EQ(static_output_shapes[0], StaticShape({20, 1, 48}));
}
