// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include <one_hot_shape_inference.hpp>
#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, MatMulTest) {
    auto A_input = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{-1, -1, -1});
    auto B_input = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{-1, -1, -1});

    auto matmul = std::make_shared<op::v0::MatMul>(A_input, B_input, 0, 1);
    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 5, 7}, StaticShape{3, 6, 7}},
        static_output_shapes = {StaticShape{}};
    shape_inference(matmul.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], (StaticShape{3, 5, 6}));
}