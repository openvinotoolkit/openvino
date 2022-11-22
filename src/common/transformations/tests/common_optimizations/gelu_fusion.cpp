// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include <gtest/gtest.h>
#include <math.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset9.hpp>
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

TEST_F(TransformationTestsF, GeluFusionPatternFour) {
    {
        auto data =
            std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});

        auto mul1_const =
            opset9::Constant::create(element::f32, Shape{1}, {1.0f / M_SQRT2});
        auto add_const =
            opset9::Constant::create(element::f32, Shape{1}, {0.5f});
        auto mul2_const =
            opset9::Constant::create(element::f32, Shape{1}, {0.5f});

        auto mul1 = std::make_shared<opset9::Multiply>(data, mul1_const);
        auto erf = std::make_shared<opset9::Erf>(mul1);
        auto mul2 = std::make_shared<opset9::Multiply>(erf, mul2_const);
        auto add = std::make_shared<opset9::Add>(mul2, add_const);
        auto mul3 = std::make_shared<opset9::Multiply>(data, add);

        function = std::make_shared<Function>(NodeVector{mul3}, ParameterVector{data});

        manager.register_pass<ov::pass::GeluFusionWithErfFour>();
    }

    {
        auto data =
            std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<opset9::Gelu>(data);
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

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_equal_const_values) {
    {
        auto input = std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{3.0f});
        auto pow = std::make_shared<opset9::Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.044715f});
        auto mul_0 = std::make_shared<opset9::Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<opset9::Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(2.0f / pi)});
        auto mul_1 =  std::make_shared<opset9::Multiply>(add_0, mul_1_constant);

        auto tanh =  std::make_shared<opset9::Tanh>(mul_1);

        auto add_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{1.0f});
        auto add_1 =  std::make_shared<opset9::Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.5f});
        auto mul_2 =  std::make_shared<opset9::Multiply>(add_1, mul_2_constant);

        auto mul_3 =  std::make_shared<opset9::Multiply>(input, mul_2);

        function = std::make_shared<Function>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<pass::GeluFusionWithTanh>();
    }

    {
        auto data =
            std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<opset9::Gelu>(data,  op::GeluApproximationMode::TANH);
        function_ref =
            std::make_shared<Function>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_params_no_conversion) {
    {
        auto input = std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});
        auto pow_param = std::make_shared<opset9::Parameter>(element::f32, Shape{1});
        auto pow = std::make_shared<opset9::Power>(input, pow_param);
        auto mul_0_param = std::make_shared<opset9::Parameter>(element::f32, Shape{1});
        auto mul_0 = std::make_shared<opset9::Multiply>(pow, mul_0_param);
        auto add_0 = std::make_shared<opset9::Add>(input, mul_0);

        auto mul_1_param =  std::make_shared<opset9::Parameter>(element::f32, Shape{1});
        auto mul_1 =  std::make_shared<opset9::Multiply>(add_0, mul_1_param);

        auto tanh =  std::make_shared<opset9::Tanh>(mul_1);

        auto add_1_param =  std::make_shared<opset9::Parameter>(element::f32, Shape{1});
        auto add_1 =  std::make_shared<opset9::Add>(tanh, add_1_param);

        auto mul_2_param = std::make_shared<opset9::Parameter>(element::f32, Shape{1});
        auto mul_2 =  std::make_shared<opset9::Multiply>(add_1, mul_2_param);

        auto mul_3 =  std::make_shared<opset9::Multiply>(input, mul_2);

        function = std::make_shared<Function>(NodeVector{mul_3},
                                              ParameterVector{input, pow_param, mul_0_param, mul_1_param, add_1_param, mul_2_param});
        manager.register_pass<pass::GeluFusionWithTanh>();
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_epsilon_pow_value) {
    {
        auto input = std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{3.0f + 1.0e-8f});
        auto pow = std::make_shared<opset9::Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.044715f});
        auto mul_0 = std::make_shared<opset9::Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<opset9::Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(2.0f / pi)});
        auto mul_1 =  std::make_shared<opset9::Multiply>(add_0, mul_1_constant);

        auto tanh =  std::make_shared<opset9::Tanh>(mul_1);

        auto add_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{1.0f});
        auto add_1 =  std::make_shared<opset9::Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.5f});
        auto mul_2 =  std::make_shared<opset9::Multiply>(add_1, mul_2_constant);

        auto mul_3 =  std::make_shared<opset9::Multiply>(input, mul_2);

        function = std::make_shared<Function>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<pass::GeluFusionWithTanh>();
    }

    {
        auto data =
            std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<opset9::Gelu>(data,  op::GeluApproximationMode::TANH);
        function_ref =
            std::make_shared<Function>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_wrong_pow_value) {
    {
        auto input = std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{2.0f});
        auto pow = std::make_shared<opset9::Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.044715f});
        auto mul_0 = std::make_shared<opset9::Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<opset9::Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(2.0f / pi)});
        auto mul_1 =  std::make_shared<opset9::Multiply>(add_0, mul_1_constant);

        auto tanh =  std::make_shared<opset9::Tanh>(mul_1);

        auto add_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{1.0f});
        auto add_1 =  std::make_shared<opset9::Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.5f});
        auto mul_2 =  std::make_shared<opset9::Multiply>(add_1, mul_2_constant);

        auto mul_3 =  std::make_shared<opset9::Multiply>(input, mul_2);

        function = std::make_shared<Function>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<pass::GeluFusionWithTanh>();
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_epsilon_mul_0_value) {
    {
        auto input = std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{3.0f});
        auto pow = std::make_shared<opset9::Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.04515f});
        auto mul_0 = std::make_shared<opset9::Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<opset9::Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(2.0f / pi)});
        auto mul_1 =  std::make_shared<opset9::Multiply>(add_0, mul_1_constant);

        auto tanh =  std::make_shared<opset9::Tanh>(mul_1);

        auto add_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{1.0f});
        auto add_1 =  std::make_shared<opset9::Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.5f});
        auto mul_2 =  std::make_shared<opset9::Multiply>(add_1, mul_2_constant);

        auto mul_3 =  std::make_shared<opset9::Multiply>(input, mul_2);

        function = std::make_shared<Function>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<pass::GeluFusionWithTanh>();
    }

    {
        auto data =
            std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<opset9::Gelu>(data,  op::GeluApproximationMode::TANH);
        function_ref =
            std::make_shared<Function>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_wrong_mul_0_value) {
    {
        auto input = std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{3.0f});
        auto pow = std::make_shared<opset9::Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{1.4715f});
        auto mul_0 = std::make_shared<opset9::Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<opset9::Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(2.0f / pi)});
        auto mul_1 =  std::make_shared<opset9::Multiply>(add_0, mul_1_constant);

        auto tanh =  std::make_shared<opset9::Tanh>(mul_1);

        auto add_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{1.0f});
        auto add_1 =  std::make_shared<opset9::Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.5f});
        auto mul_2 =  std::make_shared<opset9::Multiply>(add_1, mul_2_constant);

        auto mul_3 =  std::make_shared<opset9::Multiply>(input, mul_2);

        function = std::make_shared<Function>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<pass::GeluFusionWithTanh>();
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_epsilon_mul_1_value) {
    {
        auto input = std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{3.0f});
        auto pow = std::make_shared<opset9::Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.044715f});
        auto mul_0 = std::make_shared<opset9::Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<opset9::Add>(input, mul_0);

        auto mul_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.7980868f});
        auto mul_1 =  std::make_shared<opset9::Multiply>(add_0, mul_1_constant);

        auto tanh =  std::make_shared<opset9::Tanh>(mul_1);

        auto add_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{1.0f});
        auto add_1 =  std::make_shared<opset9::Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.5f});
        auto mul_2 =  std::make_shared<opset9::Multiply>(add_1, mul_2_constant);

        auto mul_3 =  std::make_shared<opset9::Multiply>(input, mul_2);

        function = std::make_shared<Function>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<pass::GeluFusionWithTanh>();
    }

    {
        auto data =
            std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<opset9::Gelu>(data, op::GeluApproximationMode::TANH);
        function_ref =
            std::make_shared<Function>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_wrong_mul_1_value) {
    {
        auto input = std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{3.0f});
        auto pow = std::make_shared<opset9::Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.044715f});
        auto mul_0 = std::make_shared<opset9::Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<opset9::Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(10.0f / pi)});
        auto mul_1 =  std::make_shared<opset9::Multiply>(add_0, mul_1_constant);

        auto tanh =  std::make_shared<opset9::Tanh>(mul_1);

        auto add_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{1.0f});
        auto add_1 =  std::make_shared<opset9::Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.5f});
        auto mul_2 =  std::make_shared<opset9::Multiply>(add_1, mul_2_constant);

        auto mul_3 =  std::make_shared<opset9::Multiply>(input, mul_2);

        function = std::make_shared<Function>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<pass::GeluFusionWithTanh>();
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_epsilon_add_1_value) {
    {
        auto input = std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{3.0f});
        auto pow = std::make_shared<opset9::Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.044715f});
        auto mul_0 = std::make_shared<opset9::Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<opset9::Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(2.0f / pi)});
        auto mul_1 =  std::make_shared<opset9::Multiply>(add_0, mul_1_constant);

        auto tanh =  std::make_shared<opset9::Tanh>(mul_1);

        auto add_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{1.0f + 1.0e-8f});
        auto add_1 =  std::make_shared<opset9::Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.5f});
        auto mul_2 =  std::make_shared<opset9::Multiply>(add_1, mul_2_constant);

        auto mul_3 =  std::make_shared<opset9::Multiply>(input, mul_2);

        function = std::make_shared<Function>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<pass::GeluFusionWithTanh>();
    }

    {
        auto data =
            std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<opset9::Gelu>(data,  op::GeluApproximationMode::TANH);
        function_ref =
            std::make_shared<Function>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_wrong_add_1_value) {
    {
        auto input = std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{3.0f});
        auto pow = std::make_shared<opset9::Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.044715f});
        auto mul_0 = std::make_shared<opset9::Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<opset9::Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(2.0f / pi)});
        auto mul_1 =  std::make_shared<opset9::Multiply>(add_0, mul_1_constant);

        auto tanh =  std::make_shared<opset9::Tanh>(mul_1);

        auto add_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{2.0f});
        auto add_1 =  std::make_shared<opset9::Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.5f});
        auto mul_2 =  std::make_shared<opset9::Multiply>(add_1, mul_2_constant);

        auto mul_3 =  std::make_shared<opset9::Multiply>(input, mul_2);

        function = std::make_shared<Function>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<pass::GeluFusionWithTanh>();
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_epsilon_mul_2_value) {
    {
        auto input = std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{3.0f});
        auto pow = std::make_shared<opset9::Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.044715f});
        auto mul_0 = std::make_shared<opset9::Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<opset9::Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(2.0f / pi)});
        auto mul_1 =  std::make_shared<opset9::Multiply>(add_0, mul_1_constant);

        auto tanh =  std::make_shared<opset9::Tanh>(mul_1);

        auto add_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{1.0f});
        auto add_1 =  std::make_shared<opset9::Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.5f + 1.0e-8f});
        auto mul_2 =  std::make_shared<opset9::Multiply>(add_1, mul_2_constant);

        auto mul_3 =  std::make_shared<opset9::Multiply>(input, mul_2);

        function = std::make_shared<Function>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<pass::GeluFusionWithTanh>();
    }

    {
        auto data =
            std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<opset9::Gelu>(data,  op::GeluApproximationMode::TANH);
        function_ref =
            std::make_shared<Function>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_wrong_mul_2_value) {
    {
        auto input = std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{3.0f});
        auto pow = std::make_shared<opset9::Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{0.044715f});
        auto mul_0 = std::make_shared<opset9::Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<opset9::Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(2.0f / pi)});
        auto mul_1 =  std::make_shared<opset9::Multiply>(add_0, mul_1_constant);

        auto tanh =  std::make_shared<opset9::Tanh>(mul_1);

        auto add_1_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{1.0f});
        auto add_1 =  std::make_shared<opset9::Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<opset9::Constant>(element::f32, Shape{1}, std::vector<float>{5.0f});
        auto mul_2 =  std::make_shared<opset9::Multiply>(add_1, mul_2_constant);

        auto mul_3 =  std::make_shared<opset9::Multiply>(input, mul_2);

        function = std::make_shared<Function>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<pass::GeluFusionWithTanh>();
    }
}
