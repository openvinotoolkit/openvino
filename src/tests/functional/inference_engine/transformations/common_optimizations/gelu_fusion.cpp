// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include <gtest/gtest.h>
#include <math.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <queue>
#include <string>
#include <transformations/common_optimizations/gelu_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, GeluFusionPatternOne) {
    {
        auto data =
            std::make_shared<opset7::Parameter>(element::f32, Shape{2, 2});

        auto div_const =
            opset7::Constant::create(element::f32, Shape{1}, {M_SQRT2});
        auto add_const =
            opset7::Constant::create(element::f32, Shape{1}, {1.0});
        auto mul_const =
            opset7::Constant::create(element::f32, Shape{1}, {0.5});

        auto div = std::make_shared<opset7::Divide>(data, div_const);
        auto erf = std::make_shared<opset7::Erf>(div);
        auto add = std::make_shared<opset7::Add>(erf, add_const);
        auto mul_first = std::make_shared<opset7::Multiply>(data, mul_const);
        auto mul = std::make_shared<opset7::Multiply>(mul_first, add);

        function = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<pass::GeluFusionWithErfOne>();
    }

    {
        auto data =
            std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<opset7::Gelu>(data);
        function_ref =
            std::make_shared<Function>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionPatternTwo) {
    {
        auto data =
            std::make_shared<opset7::Parameter>(element::f32, Shape{2, 2});

        auto div_const =
            opset7::Constant::create(element::f32, Shape{1}, {M_SQRT2});
        auto add_const =
            opset7::Constant::create(element::f32, Shape{1}, {1.0});
        auto mul_const =
            opset7::Constant::create(element::f32, Shape{1}, {0.5});

        auto div = std::make_shared<opset7::Divide>(data, div_const);
        auto erf = std::make_shared<opset7::Erf>(div);
        auto add = std::make_shared<opset7::Add>(erf, add_const);
        auto mul_first = std::make_shared<opset7::Multiply>(data, add);
        auto mul = std::make_shared<opset7::Multiply>(mul_first, mul_const);

        function = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<pass::GeluFusionWithErfTwo>();
    }

    {
        auto data =
            std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<opset7::Gelu>(data);
        function_ref =
            std::make_shared<Function>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionPatternThree) {
    {
        auto data =
            std::make_shared<opset7::Parameter>(element::f32, Shape{2, 2});

        auto div_const =
            opset7::Constant::create(element::f32, Shape{1}, {M_SQRT2});
        auto add_const =
            opset7::Constant::create(element::f32, Shape{1}, {1.0});
        auto mul_const =
            opset7::Constant::create(element::f32, Shape{1}, {0.5});

        auto div = std::make_shared<opset7::Divide>(data, div_const);
        auto erf = std::make_shared<opset7::Erf>(div);
        auto add = std::make_shared<opset7::Add>(erf, add_const);
        auto mul_first = std::make_shared<opset7::Multiply>(add, mul_const);
        auto mul = std::make_shared<opset7::Multiply>(data, mul_first);

        function = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<pass::GeluFusionWithErfThree>();
    }

    {
        auto data =
            std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<opset7::Gelu>(data);
        function_ref =
            std::make_shared<Function>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionPatternIncorrectDivConstValue) {
    {
        auto data =
            std::make_shared<opset7::Parameter>(element::f32, Shape{2, 2});

        auto div_const =
            opset7::Constant::create(element::f32, Shape{1}, {1.4149});
        auto add_const =
            opset7::Constant::create(element::f32, Shape{1}, {1.0});
        auto mul_const =
            opset7::Constant::create(element::f32, Shape{1}, {0.5});

        auto div = std::make_shared<opset7::Divide>(data, div_const);
        auto erf = std::make_shared<opset7::Erf>(div);
        auto add = std::make_shared<opset7::Add>(erf, add_const);
        auto mul_first = std::make_shared<opset7::Multiply>(data, add);
        auto mul = std::make_shared<opset7::Multiply>(mul_first, mul_const);

        function = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});
        function_ref =
            std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<pass::GeluFusionWithErfTwo>();
    }
}

TEST_F(TransformationTestsF, GeluFusionPatternTooShortDivConstValue) {
    {
        auto data =
            std::make_shared<opset7::Parameter>(element::f32, Shape{2, 2});

        auto div_const =
            opset7::Constant::create(element::f32, Shape{1}, {1.4142});
        auto add_const =
            opset7::Constant::create(element::f32, Shape{1}, {1.0});
        auto mul_const =
            opset7::Constant::create(element::f32, Shape{1}, {0.5});

        auto div = std::make_shared<opset7::Divide>(data, div_const);
        auto erf = std::make_shared<opset7::Erf>(div);
        auto add = std::make_shared<opset7::Add>(erf, add_const);
        auto mul_first = std::make_shared<opset7::Multiply>(data, add);
        auto mul = std::make_shared<opset7::Multiply>(mul_first, mul_const);

        function = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});
        function_ref =
            std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<pass::GeluFusionWithErfTwo>();
    }
}
