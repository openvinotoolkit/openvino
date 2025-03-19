// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/leaky_relu_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

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

TEST_F(TransformationTestsF, LeakyReluFusionConstantGreaterThanOne) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 2});
        auto alpha = opset8::Constant::create(element::f32, Shape{1}, {1.1});
        auto multiply = std::make_shared<opset8::Multiply>(data, alpha);
        auto max = std::make_shared<opset8::Maximum>(data, multiply);
        model = std::make_shared<Model>(NodeVector{max}, ParameterVector{data});

        manager.register_pass<ov::pass::LeakyReluFusion>();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, LeakyReluFusionConstantAlphaOnFirstInput) {
    {
        auto alpha = opset8::Constant::create(element::f32, Shape{1}, {0.1});
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 2});
        auto multiply = std::make_shared<opset8::Multiply>(alpha, data);
        auto max = std::make_shared<opset8::Maximum>(multiply, data);
        model = std::make_shared<Model>(NodeVector{max}, ParameterVector{data});

        manager.register_pass<ov::pass::LeakyReluFusion>();
    }

    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
        auto alpha = opset8::Constant::create(element::f32, Shape{1}, {0.1});
        auto leaky_relu = std::make_shared<opset8::PRelu>(data, alpha);
        model_ref = std::make_shared<Model>(NodeVector{leaky_relu}, ParameterVector{data});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
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
}
