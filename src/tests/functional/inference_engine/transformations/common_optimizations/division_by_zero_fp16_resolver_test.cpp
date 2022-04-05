// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <openvino/opsets/opset4.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/division_by_zero_fp16_resolver.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov;
constexpr float normalized_fp16_min = 6.103515625e-05f;  // fp16 minimal normalized value


TEST_F(TransformationTestsF, DivisionByZeroMinimalPattern) {
    const float eps_value = 1.e-12;
    {
        auto input_1 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset4::Add>(input_2, eps_const);
        auto divide = std::make_shared<opset4::Divide>(input_1, add);

        function = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::DivisionByZeroFP16Resolver>();
    }

    {
        auto input_1 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {normalized_fp16_min});
        auto add = std::make_shared<opset4::Add>(input_2, eps_const);
        auto divide = std::make_shared<opset4::Divide>(input_1, add);

        function_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, PowWithNegativeExponent) {
    const float eps_value = 1.e-12;
    {
        auto input_1 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset4::Add>(input_2, eps_const);
        auto pow_exp_const = opset4::Constant::create(element::f32, Shape{1}, {-1.77});
        auto pow = std::make_shared<opset4::Power>(add, pow_exp_const);
        auto mul = std::make_shared<opset4::Multiply>(input_1, pow);

        function = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::DivisionByZeroFP16Resolver>();
    }

    {
        auto input_1 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {normalized_fp16_min});
        auto add = std::make_shared<opset4::Add>(input_2, eps_const);
        auto pow_exp_const = opset4::Constant::create(element::f32, Shape{1}, {-1.77});
        auto pow = std::make_shared<opset4::Power>(add, pow_exp_const);
        auto mul = std::make_shared<opset4::Multiply>(input_1, pow);

        function_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, PowWithPositiveExponent) {
    // graph should be left unchanged
    const float eps_value = 1.e-12;
    {
        auto input_1 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset4::Add>(input_2, eps_const);
        auto pow_exp_const = opset4::Constant::create(element::f32, Shape{1}, {1.77});
        auto pow = std::make_shared<opset4::Power>(add, pow_exp_const);
        auto mul = std::make_shared<opset4::Multiply>(input_1, pow);

        function = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::DivisionByZeroFP16Resolver>();
    }

    {
        auto input_1 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset4::Add>(input_2, eps_const);
        auto pow_exp_const = opset4::Constant::create(element::f32, Shape{1}, {1.77});
        auto pow = std::make_shared<opset4::Power>(add, pow_exp_const);
        auto mul = std::make_shared<opset4::Multiply>(input_1, pow);

        function_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, DivisionByZeroMinimalPatternUnchanged) {
    // if eps_value is greater than normalized_fp16_min then leave graph unchanged
    const float eps_value = 0.0001f;
    {
        auto input_1 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset4::Add>(input_2, eps_const);
        auto divide = std::make_shared<opset4::Divide>(input_1, add);

        function = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::DivisionByZeroFP16Resolver>();
    }

    {
        auto input_1 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset4::Add>(input_2, eps_const);
        auto divide = std::make_shared<opset4::Divide>(input_1, add);

        function_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, DivisionByZeroInL2NormWithSqrtAndWithMax) {
    const float eps_value = 1.e-12;
    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = opset4::Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<opset4::Power>(input, exp);
        auto axes_const = opset4::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset4::ReduceSum>(pow, axes_const);
        auto eps_const = opset4::Constant::create(element::f32, Shape{}, {eps_value});
        auto max = std::make_shared<opset4::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset4::Sqrt>(max);
        auto divide = std::make_shared<opset4::Divide>(input, sqrt);

        function = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});

        manager.register_pass<pass::DivisionByZeroFP16Resolver>();
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = opset4::Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<opset4::Power>(input, exp);
        auto axes_const = opset4::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset4::ReduceSum>(pow, axes_const);
        auto eps_const = opset4::Constant::create(element::f32, Shape{}, {normalized_fp16_min});
        auto max = std::make_shared<opset4::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset4::Sqrt>(max);
        auto divide = std::make_shared<opset4::Divide>(input, sqrt);

        function_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}


TEST_F(TransformationTestsF, DivisionByZeroInL2NormWithSqrtAndWithAdd) {
    const float eps_value = 1.e-12;
    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = opset4::Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<opset4::Power>(input, exp);
        auto axes_const = opset4::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset4::ReduceSum>(pow, axes_const);
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset4::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset4::Sqrt>(add);
        auto divide = std::make_shared<opset4::Divide>(input, sqrt);

        function = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});

        manager.register_pass<pass::DivisionByZeroFP16Resolver>();
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = opset4::Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<opset4::Power>(input, exp);
        auto axes_const = opset4::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset4::ReduceSum>(pow, axes_const);
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {normalized_fp16_min});
        auto add = std::make_shared<opset4::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset4::Sqrt>(add);
        auto divide = std::make_shared<opset4::Divide>(input, sqrt);

        function_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}
