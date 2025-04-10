// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/normalize_l2_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, NormalizeL2FusionWithMax) {
    const float eps_value = 0.000099f;
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(3));
        auto exp = opset4::Constant::create(element::f16, Shape{}, {2.f});
        auto pow = std::make_shared<opset4::Power>(input, exp);
        auto axes_const = opset4::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset4::ReduceSum>(pow, axes_const);
        auto eps_const = opset4::Constant::create(element::f16, Shape{}, {eps_value});
        auto max = std::make_shared<opset4::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset4::Sqrt>(max);
        auto divide = std::make_shared<opset4::Divide>(input, sqrt);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{input});
        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(3));
        auto axes_const = opset4::Constant::create(element::i64, Shape{2}, {0, 1});
        auto normalize_l2 = std::make_shared<opset4::NormalizeL2>(input, axes_const, eps_value, op::EpsMode::MAX);

        model_ref = std::make_shared<ov::Model>(NodeVector{normalize_l2}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithMaxIncorrectExp) {
    const float eps_value = 0.0009f;
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(2));
        auto exp = opset4::Constant::create(element::f16, Shape{}, {3.f});
        auto pow = std::make_shared<opset4::Power>(input, exp);
        auto axes_const = opset4::Constant::create(element::i64, Shape{1}, {0});
        auto reduce_sum = std::make_shared<opset4::ReduceSum>(pow, axes_const);
        auto eps_const = opset4::Constant::create(element::f16, Shape{}, {eps_value});
        auto max = std::make_shared<opset4::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset4::Sqrt>(max);
        auto divide = std::make_shared<opset4::Divide>(input, sqrt);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{input});

        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithMaxIncorrectEpsValueShape) {
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(2));
        auto exp = opset4::Constant::create(element::f16, Shape{}, {2.f});
        auto pow = std::make_shared<opset4::Power>(input, exp);
        auto axes_const = opset4::Constant::create(element::i64, Shape{1}, {0});
        auto reduce_sum = std::make_shared<opset4::ReduceSum>(pow, axes_const);
        auto eps_const = opset4::Constant::create(element::f16, Shape{2}, {1, 2});
        auto max = std::make_shared<opset4::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset4::Sqrt>(max);
        auto divide = std::make_shared<opset4::Divide>(input, sqrt);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{input});

        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithAdd) {
    const float eps_value = 0.000099f;
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

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{input});

        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto axes_const = opset4::Constant::create(element::i64, Shape{2}, {0, 1});
        auto normalize_l2 = std::make_shared<opset4::NormalizeL2>(input, axes_const, eps_value, op::EpsMode::ADD);

        model_ref = std::make_shared<ov::Model>(NodeVector{normalize_l2}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithAddIncorrectExp) {
    const float eps_value = 0.0009f;
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(2));
        auto exp = opset4::Constant::create(element::f16, Shape{}, {1.9f});
        auto pow = std::make_shared<opset4::Power>(input, exp);
        auto axes_const = opset4::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset4::ReduceSum>(pow, axes_const);
        auto eps_const = opset4::Constant::create(element::f16, Shape{}, {eps_value});
        auto add = std::make_shared<opset4::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset4::Sqrt>(add);
        auto divide = std::make_shared<opset4::Divide>(input, sqrt);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{input});
        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithAddIncorrectEpsValueShape) {
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(4));
        auto exp = opset4::Constant::create(element::f16, Shape{}, {2.f});
        auto pow = std::make_shared<opset4::Power>(input, exp);
        auto axes_const = opset4::Constant::create(element::i64, Shape{1}, {0});
        auto reduce_sum = std::make_shared<opset4::ReduceSum>(pow, axes_const);
        auto eps_const = opset4::Constant::create(element::f16, Shape{2}, {1, 2});
        auto add = std::make_shared<opset4::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset4::Sqrt>(add);
        auto divide = std::make_shared<opset4::Divide>(input, sqrt);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{input});
        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithMaxMul) {
    const float eps_value = 0.000099f;
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 2, 3});
        auto exp = opset4::Constant::create(element::f16, Shape{}, {2.f});
        auto pow = std::make_shared<opset4::Power>(input, exp);
        auto axes_const = opset4::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset4::ReduceSum>(pow, axes_const);
        auto eps_const = opset4::Constant::create(element::f16, Shape{}, {eps_value});
        auto max = std::make_shared<opset4::Maximum>(reduce_sum, eps_const);
        auto power_const = opset4::Constant::create(element::f16, Shape{}, {-0.5f});
        auto unsqrt = std::make_shared<opset8::Power>(max, power_const);
        auto mul = std::make_shared<opset4::Multiply>(input, unsqrt);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});
        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 2, 3});
        auto axes_const = opset4::Constant::create(element::i64, Shape{2}, {0, 1});
        auto normalize_l2 = std::make_shared<opset4::NormalizeL2>(input, axes_const, eps_value, op::EpsMode::MAX);

        model_ref = std::make_shared<ov::Model>(NodeVector{normalize_l2}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithMaxMulIncorrectSecondExp) {
    const float eps_value = 0.000099f;
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(3));
        auto exp = opset4::Constant::create(element::f16, Shape{}, {2.f});
        auto pow = std::make_shared<opset4::Power>(input, exp);
        auto axes_const = opset4::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset4::ReduceSum>(pow, axes_const);
        auto eps_const = opset4::Constant::create(element::f16, Shape{}, {eps_value});
        auto max = std::make_shared<opset4::Maximum>(reduce_sum, eps_const);
        auto power_const = opset4::Constant::create(element::f16, Shape{}, {-0.6f});
        auto unsqrt = std::make_shared<opset8::Power>(max, power_const);
        auto mul = std::make_shared<opset4::Multiply>(input, unsqrt);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});

        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithMaxSqrtAsPower) {
    const float eps_value = 0.000099f;
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(3));
        auto exp = opset4::Constant::create(element::f16, Shape{}, {2.f});
        auto pow = std::make_shared<opset4::Power>(input, exp);
        auto axes_const = opset4::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset4::ReduceSum>(pow, axes_const);
        auto eps_const = opset4::Constant::create(element::f16, Shape{}, {eps_value});
        auto max = std::make_shared<opset4::Maximum>(reduce_sum, eps_const);
        auto sqrt_exp = opset4::Constant::create(element::f16, Shape{}, {0.5f});
        auto sqrt = std::make_shared<opset4::Power>(max, sqrt_exp);
        auto divide = std::make_shared<opset4::Divide>(input, sqrt);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{input});
        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(3));
        auto axes_const = opset4::Constant::create(element::i64, Shape{2}, {0, 1});
        auto normalize_l2 = std::make_shared<opset4::NormalizeL2>(input, axes_const, eps_value, op::EpsMode::MAX);

        model_ref = std::make_shared<ov::Model>(NodeVector{normalize_l2}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithMaxSqrtAsPowerIncorrectPowerExp) {
    const float eps_value = 0.0009f;
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(3));
        auto exp = opset4::Constant::create(element::f16, Shape{}, {2.f});
        auto pow = std::make_shared<opset4::Power>(input, exp);
        auto axes_const = opset4::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset4::ReduceSum>(pow, axes_const);
        auto eps_const = opset4::Constant::create(element::f16, Shape{}, {eps_value});
        auto max = std::make_shared<opset4::Maximum>(reduce_sum, eps_const);
        auto sqrt_exp = opset4::Constant::create(element::f16, Shape{}, {0.9f});
        auto sqrt = std::make_shared<opset4::Power>(max, sqrt_exp);
        auto divide = std::make_shared<opset4::Divide>(input, sqrt);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{input});

        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }
}
