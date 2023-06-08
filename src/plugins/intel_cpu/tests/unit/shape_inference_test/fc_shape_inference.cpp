// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "transformations/cpu_opset/common/op/fully_connected.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, FC_InputSize_2) {
    auto activate = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1 });
    auto weight = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{5, 6});
    auto op = std::make_shared<ov::intel_cpu::FullyConnectedNode>(activate, weight, ngraph::Rank(5), element::f32);
    std::vector<StaticShape> static_input_shapes = {StaticShape{720, 640}, {5, 6}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 1, 1, 720, 5}};
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TEST(StaticShapeInferenceTest, FC_InputSize_3) {
    auto activate = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto weight = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{5, 6});
    auto channel = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{5});
    auto op = std::make_shared<ov::intel_cpu::FullyConnectedNode>(activate, weight, channel, ngraph::Rank(5), element::f32);
    std::vector<StaticShape> static_input_shapes = {StaticShape{720, 640}, {5, 6}, {5}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 1, 1, 720, 5}};
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

