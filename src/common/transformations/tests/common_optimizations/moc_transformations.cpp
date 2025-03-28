// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/moc_transformations.hpp"

#include <gtest/gtest.h>

#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"

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

TEST(TransformationTests, MOCConvertElimination) {
    auto input = std::make_shared<opset12::Parameter>(element::f32, Shape{1});
    auto const_val = opset12::Constant::create(element::f32, Shape{1}, {2});

    auto add1 = std::make_shared<opset12::Add>(input, const_val);
    auto convert_fp32 = std::make_shared<opset12::Convert>(const_val, element::f32);
    auto mul = std::make_shared<opset12::MatMul>(add1, convert_fp32);

    auto model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input});
    ov::pass::Manager m;
    m.register_pass<ov::pass::MOCTransformations>(false);
    m.run_passes(model);

    EXPECT_EQ(count_ops_of_type<opset12::Constant>(model), 1);
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

TEST_F(TransformationTestsF, SqueezeRemainsSqueezeAfterMOC) {
    {
        using namespace ov::op;
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{30});
        auto shape = v0::Constant::create(element::i64, Shape{5}, {2, 3, 1, 5, 1});
        auto reshape = std::make_shared<v1::Reshape>(input, shape, false);
        auto unsqueeze_axes = v0::Constant::create(element::i64, Shape{1}, {0});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(reshape, unsqueeze_axes);

        auto squeeze_axes = v0::Constant::create(element::i64, Shape{2}, {3, 5});
        auto squeeze = std::make_shared<v0::Squeeze>(unsqueeze, squeeze_axes);

        auto res = std::make_shared<v0::Result>(squeeze);
        model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MOCTransformations>(false);
    }
}

TEST_F(TransformationTestsF, MOCTest) {
    std::shared_ptr<ov::Node> weights;
    std::shared_ptr<ov::Node> weights_ref;
    {
        using namespace ov::op;
        auto data = std::make_shared<v0::Parameter>(element::f32, ov::PartialShape{-1, -1, 3840});
        auto data1 = std::make_shared<v0::Parameter>(element::f32, ov::PartialShape{-1, -1, 3840});
        auto a_mul = std::make_shared<v1::Multiply>(data, data1);
        weights = std::make_shared<v0::Constant>(element::f32, ov::Shape{1536, 3840});
        auto scale = std::make_shared<v0::Constant>(element::f32, ov::Shape{}, 0.194145);
        auto matmul = std::make_shared<v0::MatMul>(a_mul, weights, false, true);
        auto mul = std::make_shared<v1::Multiply>(matmul, scale);
        auto res = std::make_shared<v0::Result>(mul);

        weights->set_friendly_name("self.model.layers.50.mlp.down_proj.weight");
        ov::disable_constant_folding(matmul);

        model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{data, data1});
        manager.register_pass<ov::pass::MOCTransformations>(false);
    }
    {
        using namespace ov::op;
        auto data = std::make_shared<v0::Parameter>(element::f32, ov::PartialShape{-1, -1, 3840});
        auto data1 = std::make_shared<v0::Parameter>(element::f32, ov::PartialShape{-1, -1, 3840});
        auto a_mul = std::make_shared<v1::Multiply>(data, data1);
        weights_ref = std::make_shared<v0::Constant>(element::f32, ov::Shape{1536, 3840});
        auto scale = std::make_shared<v0::Constant>(element::f32, ov::Shape{1, 1, 1}, 0.194145);
        auto matmul = std::make_shared<v0::MatMul>(a_mul, weights_ref, false, true);
        auto mul = std::make_shared<v1::Multiply>(matmul, scale);
        auto res = std::make_shared<v0::Result>(mul);

        weights_ref->set_friendly_name("self.model.layers.50.mlp.down_proj.weight");
        ov::disable_constant_folding(matmul);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{data, data1});

        EXPECT_EQ(weights->get_friendly_name(), weights_ref->get_friendly_name());
    }
}