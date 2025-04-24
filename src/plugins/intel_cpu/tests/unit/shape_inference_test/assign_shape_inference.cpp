// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

template <class T>
std::shared_ptr<T> constructGraph();

template <>
std::shared_ptr<op::v3::Assign> constructGraph() {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto read_value = std::make_shared<op::v3::ReadValue>(input, "variable_id");
    return std::make_shared<op::v3::Assign>(read_value, "variable_id");
}

template <>
std::shared_ptr<op::v6::Assign> constructGraph() {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{PartialShape::dynamic(), element::dynamic, "ID"});
    auto read_value = std::make_shared<op::v6::Assign>(input, variable);
    return std::make_shared<op::v6::Assign>(read_value, variable);
}

template <class T>
void assignTest() {
    auto assign = constructGraph<T>();

    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 2, 64, 64}}, static_output_shapes = {StaticShape{}};
    static_output_shapes = shape_inference(assign.get(), static_input_shapes);
    ASSERT_EQ(static_input_shapes[0], (StaticShape{1, 2, 64, 64}));
}

TEST(StaticShapeInferenceTest, AssignTest) {
    // Test v3 Assign
    assignTest<op::v3::Assign>();
    // Test v6 Assign
    assignTest<op::v6::Assign>();
}
