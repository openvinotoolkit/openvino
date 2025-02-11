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
std::shared_ptr<op::v3::ReadValue> constructGraph() {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    return std::make_shared<op::v3::ReadValue>(input, "variable_id");
}

template <>
std::shared_ptr<op::v6::ReadValue> constructGraph() {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{PartialShape::dynamic(), element::dynamic, "ID"});
    return std::make_shared<op::v6::ReadValue>(input, variable);
}

template <class T>
void readValueTest() {
    auto readValue = constructGraph<T>();

    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 2, 64, 64}};
    const auto static_output_shapes = shape_inference(readValue.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes[0], (StaticShape{1, 2, 64, 64}));
}

TEST(StaticShapeInferenceTest, ReadValueTest) {
    // Test v3 ReadValue
    readValueTest<op::v3::ReadValue>();
    // Test v6 ReadValue
    readValueTest<op::v6::ReadValue>();
}
