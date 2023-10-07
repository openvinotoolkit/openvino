// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/moc_transformations.hpp"

#include <gtest/gtest.h>

#include <string>

#include "openvino/core/model.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"

using namespace testing;
using namespace ov;
using namespace ov::opset12;

TEST(TransformationTests, TestModelTensorsConsistencyUseShapesTrue) {
    auto input = std::make_shared<opset12::Parameter>(element::f32, Shape{1});
    auto const1 = opset12::Constant::create(element::f32, Shape{1}, {1});
    auto const2 = opset12::Constant::create(element::f32, Shape{1}, {2});
    auto const3 = opset12::Constant::create(element::f32, Shape{1}, {3});
    auto add1 = std::make_shared<opset12::Add>(input, const1);
    auto add2 = std::make_shared<opset12::Add>(add1, const2);
    auto add3 = std::make_shared<opset12::Add>(add2, const3);

    auto model = std::make_shared<Model>(NodeVector{add3}, ParameterVector{input});
    ov::pass::Manager m;
    m.register_pass<ov::pass::MOCTransformations>(true);
    m.run_passes(model);

    std::unordered_set<std::string> new_tensors = {"new_name"};
    model->outputs()[0].set_names(new_tensors);
    EXPECT_TRUE(model->outputs()[0].get_names() == new_tensors);

    model->validate_nodes_and_infer_types();
    EXPECT_TRUE(model->outputs()[0].get_names() == new_tensors);
}

TEST(TransformationTests, TestModelTensorsConsistencyUseShapesFalse) {
    auto input = std::make_shared<opset12::Parameter>(element::f32, Shape{1});
    auto const1 = opset12::Constant::create(element::f32, Shape{1}, {1});
    auto const2 = opset12::Constant::create(element::f32, Shape{1}, {2});
    auto const3 = opset12::Constant::create(element::f32, Shape{1}, {3});
    auto add1 = std::make_shared<opset12::Add>(input, const1);
    auto add2 = std::make_shared<opset12::Add>(add1, const2);
    auto add3 = std::make_shared<opset12::Add>(add2, const3);

    auto model = std::make_shared<Model>(NodeVector{add3}, ParameterVector{input});
    ov::pass::Manager m;
    m.register_pass<ov::pass::MOCTransformations>(false);
    m.run_passes(model);

    std::unordered_set<std::string> new_tensors = {"new_name"};
    model->outputs()[0].set_names(new_tensors);
    EXPECT_TRUE(model->outputs()[0].get_names() == new_tensors);

    model->validate_nodes_and_infer_types();
    EXPECT_TRUE(model->outputs()[0].get_names() == new_tensors);
}
