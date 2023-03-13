// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include <gtest/gtest.h>
#include <math.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <openvino/opsets/opset10.hpp>
#include <transformations/common_optimizations/prelu_fusion.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, PReluFusionNegativeAdd) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 128});
        auto relu_pos = std::make_shared<ngraph::opset8::Relu>(data);
        auto neg = std::make_shared<ngraph::opset8::Negative>(data);
        auto relu_neg = std::make_shared<ngraph::opset8::Relu>(neg);
        auto neg2 = std::make_shared<ngraph::opset8::Negative>(relu_neg);
        auto mul_const = opset8::Constant::create(element::f32, Shape{1}, {0.001});
        auto mul = std::make_shared<ngraph::opset8::Multiply>(neg2, mul_const);
        auto add = std::make_shared<ngraph::opset8::Add>(relu_pos, mul);

        function = std::make_shared<Function>(NodeVector{add}, ParameterVector{data});

        manager.register_pass<ov::pass::PReluFusion>();
    }

    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 128});
        auto prelu_const = opset8::Constant::create(element::f32, Shape{1}, {0.001});
        auto prelu = std::make_shared<opset8::PRelu>(data, prelu_const);
        function_ref = std::make_shared<Function>(NodeVector{prelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, PReluFusionNegativeSub) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 128});
        auto relu_pos = std::make_shared<ngraph::opset8::Relu>(data);
        auto neg = std::make_shared<ngraph::opset8::Negative>(data);
        auto relu_neg = std::make_shared<ngraph::opset8::Relu>(neg);
        auto mul_const = opset8::Constant::create(element::f32, Shape{1}, {0.001});
        auto mul = std::make_shared<ngraph::opset8::Multiply>(relu_neg, mul_const);
        auto sub = std::make_shared<ngraph::opset8::Subtract>(relu_pos, mul);

        function = std::make_shared<Function>(NodeVector{sub}, ParameterVector{data});

        manager.register_pass<ov::pass::PReluFusion>();
    }

    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 128});
        auto prelu_const = opset8::Constant::create(element::f32, Shape{1}, {0.001});
        auto prelu = std::make_shared<opset8::PRelu>(data, prelu_const);
        function_ref = std::make_shared<Function>(NodeVector{prelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, PReluFusionMultiplyAdd) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 128});
        auto relu_pos = std::make_shared<ngraph::opset8::Relu>(data);
        auto mul_neg_const = opset8::Constant::create(element::f32, Shape{1}, {-1.0});
        auto mul_neg = std::make_shared<ngraph::opset8::Multiply>(data, mul_neg_const);
        auto relu_neg = std::make_shared<ngraph::opset8::Relu>(mul_neg);
        auto mul_const = opset8::Constant::create(element::f32, Shape{1}, {-0.001});
        auto mul = std::make_shared<ngraph::opset8::Multiply>(relu_neg, mul_const);
        auto add = std::make_shared<ngraph::opset8::Add>(relu_pos, mul);

        function = std::make_shared<Function>(NodeVector{add}, ParameterVector{data});

        manager.register_pass<ov::pass::PReluFusion>();
    }

    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 128});
        auto prelu_const = opset8::Constant::create(element::f32, Shape{1}, {0.001});
        auto prelu = std::make_shared<opset8::PRelu>(data, prelu_const);
        function_ref = std::make_shared<Function>(NodeVector{prelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, PReluFusionMultiplySub) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 128});
        auto relu_pos = std::make_shared<ngraph::opset8::Relu>(data);
        auto mul_neg_const = opset8::Constant::create(element::f32, Shape{1}, {-1.0});
        auto mul_neg = std::make_shared<ngraph::opset8::Multiply>(data, mul_neg_const);
        auto relu_neg = std::make_shared<ngraph::opset8::Relu>(mul_neg);
        auto mul_const = opset8::Constant::create(element::f32, Shape{1}, {0.001});
        auto mul = std::make_shared<ngraph::opset8::Multiply>(relu_neg, mul_const);
        auto sub = std::make_shared<ngraph::opset8::Subtract>(relu_pos, mul);

        function = std::make_shared<Function>(NodeVector{sub}, ParameterVector{data});

        manager.register_pass<ov::pass::PReluFusion>();
    }

    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 128});
        auto prelu_const = opset8::Constant::create(element::f32, Shape{1}, {0.001});
        auto prelu = std::make_shared<opset8::PRelu>(data, prelu_const);
        function_ref = std::make_shared<Function>(NodeVector{prelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, PReluFusionFail) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 128});
        auto relu_pos = std::make_shared<ngraph::opset8::Relu>(data);
        auto mul_neg_const = opset8::Constant::create(element::f32, Shape{1}, {2.0});
        auto mul_neg = std::make_shared<ngraph::opset8::Multiply>(data, mul_neg_const);
        auto relu_neg = std::make_shared<ngraph::opset8::Relu>(mul_neg);
        auto mul_const = opset8::Constant::create(element::f32, Shape{1}, {0.001});
        auto mul = std::make_shared<ngraph::opset8::Multiply>(relu_neg, mul_const);
        auto sub = std::make_shared<ngraph::opset8::Subtract>(relu_pos, mul);

        function = std::make_shared<Function>(NodeVector{sub}, ParameterVector{data});

        manager.register_pass<ov::pass::PReluFusion>();
    }

    function_ref = ngraph::clone_function(*function);
}

TEST_F(TransformationTestsF, PReluFusionAbsSubMulMulAdd) {
    using namespace std;
    using namespace ov::opset10;
    {
        const auto data = make_shared<Parameter>(element::f32, Shape{1, 128});
        const auto relu = make_shared<Relu>(data);
        const auto abs = make_shared<Abs>(data);
        const auto sub = make_shared<Subtract>(data, abs);
        const auto mul_1_const = Constant::create(element::f32, Shape{1}, {0.022});
        const auto mul_1 = make_shared<Multiply>(sub, mul_1_const);
        const auto mul_2_const = Constant::create(element::f32, Shape{1}, {0.5});
        const auto mul_2 = make_shared<Multiply>(mul_1, mul_2_const);
        const auto add = make_shared<Add>(relu, mul_2);
        function = make_shared<Function>(NodeVector{add}, ParameterVector{data});

        manager.register_pass<ov::pass::PReluFusion>();
    }
    {
        const auto data = make_shared<Parameter>(element::f32, Shape{1, 128});
        const auto prelu_const = Constant::create(element::f32, Shape{1}, {0.022});
        const auto prelu = make_shared<PRelu>(data, prelu_const);
        function_ref = make_shared<Function>(NodeVector{prelu}, ParameterVector{data});
    }
}
