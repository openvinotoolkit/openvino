// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/manager.hpp>
#include <queue>
#include <string>
#include <transformations/common_optimizations/leaky_relu_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, LeakyReluFusionConstant) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 2});
        auto alpha = opset8::Constant::create(element::f32, Shape{1}, {0.1});
        auto multiply = std::make_shared<opset8::Multiply>(data, alpha);
        auto max = std::make_shared<opset8::Maximum>(data, multiply);
        model = std::make_shared<Model>(NodeVector{max}, ParameterVector{data});

        manager.register_pass<ov::pass::LeakyReluFusion>();
    }

    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
        auto alpha = opset8::Constant::create(element::f32, Shape{1}, {0.1});
        auto leaky_relu = std::make_shared<opset8::PRelu>(data, alpha);
        model_ref = std::make_shared<Model>(NodeVector{leaky_relu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, LeakyReluFusionScalar) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 2});
        auto alpha = opset8::Constant::create(element::f32, Shape{}, {0.1});
        auto multiply = std::make_shared<opset8::Multiply>(data, alpha);
        auto max = std::make_shared<opset8::Maximum>(data, multiply);
        model = std::make_shared<Model>(NodeVector{max}, ParameterVector{data});

        manager.register_pass<ov::pass::LeakyReluFusion>();
    }

    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
        auto alpha = opset8::Constant::create(element::f32, Shape{}, {0.1});
        auto leaky_relu = std::make_shared<opset8::PRelu>(data, alpha);
        model_ref = std::make_shared<Model>(NodeVector{leaky_relu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, LeakyReluFusionParameter) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 2});
        auto alpha = std::make_shared<opset8::Parameter>(element::f32, Shape{});
        auto multiply = std::make_shared<opset8::Multiply>(data, alpha);
        auto max = std::make_shared<opset8::Maximum>(data, multiply);
        model = std::make_shared<Model>(NodeVector{max}, ParameterVector{data, alpha});

        manager.register_pass<ov::pass::LeakyReluFusion>();
    }

    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
        auto alpha = std::make_shared<opset8::Parameter>(element::f32, Shape{});
        auto leaky_relu = std::make_shared<opset8::PRelu>(data, alpha);
        model_ref = std::make_shared<Model>(NodeVector{leaky_relu}, ParameterVector{data, alpha});
    }
}
