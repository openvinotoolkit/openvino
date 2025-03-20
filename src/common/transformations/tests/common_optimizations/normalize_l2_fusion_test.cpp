// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/normalize_l2_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/sqrt.hpp"

using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, NormalizeL2FusionWithMax) {
    const float eps_value = 0.000099f;
    {
        auto input = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(3));
        auto exp = op::v0::Constant::create(element::f16, Shape{}, {2.f});
        auto pow = std::make_shared<op::v1::Power>(input, exp);
        auto axes_const = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<op::v1::ReduceSum>(pow, axes_const);
        auto eps_const = op::v0::Constant::create(element::f16, Shape{}, {eps_value});
        auto max = std::make_shared<op::v1::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<op::v0::Sqrt>(max);
        auto divide = std::make_shared<op::v1::Divide>(input, sqrt);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{input});
        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }

    {
        auto input = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(3));
        auto axes_const = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
        auto normalize_l2 = std::make_shared<op::v0::NormalizeL2>(input, axes_const, eps_value, op::EpsMode::MAX);

        model_ref = std::make_shared<ov::Model>(NodeVector{normalize_l2}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithMaxIncorrectExp) {
    const float eps_value = 0.0009f;
    {
        auto input = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(2));
        auto exp = op::v0::Constant::create(element::f16, Shape{}, {3.f});
        auto pow = std::make_shared<op::v1::Power>(input, exp);
        auto axes_const = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce_sum = std::make_shared<op::v1::ReduceSum>(pow, axes_const);
        auto eps_const = op::v0::Constant::create(element::f16, Shape{}, {eps_value});
        auto max = std::make_shared<op::v1::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<op::v0::Sqrt>(max);
        auto divide = std::make_shared<op::v1::Divide>(input, sqrt);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{input});

        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithMaxIncorrectEpsValueShape) {
    {
        auto input = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(2));
        auto exp = op::v0::Constant::create(element::f16, Shape{}, {2.f});
        auto pow = std::make_shared<op::v1::Power>(input, exp);
        auto axes_const = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce_sum = std::make_shared<op::v1::ReduceSum>(pow, axes_const);
        auto eps_const = op::v0::Constant::create(element::f16, Shape{2}, {1, 2});
        auto max = std::make_shared<op::v1::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<op::v0::Sqrt>(max);
        auto divide = std::make_shared<op::v1::Divide>(input, sqrt);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{input});

        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithAdd) {
    const float eps_value = 0.000099f;
    {
        auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = op::v0::Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<op::v1::Power>(input, exp);
        auto axes_const = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<op::v1::ReduceSum>(pow, axes_const);
        auto eps_const = op::v0::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<op::v1::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<op::v0::Sqrt>(add);
        auto divide = std::make_shared<op::v1::Divide>(input, sqrt);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{input});

        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }

    {
        auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
        auto axes_const = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
        auto normalize_l2 = std::make_shared<op::v0::NormalizeL2>(input, axes_const, eps_value, op::EpsMode::ADD);

        model_ref = std::make_shared<ov::Model>(NodeVector{normalize_l2}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithAddIncorrectExp) {
    const float eps_value = 0.0009f;
    {
        auto input = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(2));
        auto exp = op::v0::Constant::create(element::f16, Shape{}, {1.9f});
        auto pow = std::make_shared<op::v1::Power>(input, exp);
        auto axes_const = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<op::v1::ReduceSum>(pow, axes_const);
        auto eps_const = op::v0::Constant::create(element::f16, Shape{}, {eps_value});
        auto add = std::make_shared<op::v1::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<op::v0::Sqrt>(add);
        auto divide = std::make_shared<op::v1::Divide>(input, sqrt);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{input});
        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithAddIncorrectEpsValueShape) {
    {
        auto input = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
        auto exp = op::v0::Constant::create(element::f16, Shape{}, {2.f});
        auto pow = std::make_shared<op::v1::Power>(input, exp);
        auto axes_const = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce_sum = std::make_shared<op::v1::ReduceSum>(pow, axes_const);
        auto eps_const = op::v0::Constant::create(element::f16, Shape{2}, {1, 2});
        auto add = std::make_shared<op::v1::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<op::v0::Sqrt>(add);
        auto divide = std::make_shared<op::v1::Divide>(input, sqrt);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{input});
        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithMaxMul) {
    const float eps_value = 0.000099f;
    {
        auto input = std::make_shared<op::v0::Parameter>(element::f16, Shape{1, 2, 3});
        auto exp = op::v0::Constant::create(element::f16, Shape{}, {2.f});
        auto pow = std::make_shared<op::v1::Power>(input, exp);
        auto axes_const = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<op::v1::ReduceSum>(pow, axes_const);
        auto eps_const = op::v0::Constant::create(element::f16, Shape{}, {eps_value});
        auto max = std::make_shared<op::v1::Maximum>(reduce_sum, eps_const);
        auto power_const = op::v0::Constant::create(element::f16, Shape{}, {-0.5f});
        auto unsqrt = std::make_shared<op::v1::Power>(max, power_const);
        auto mul = std::make_shared<op::v1::Multiply>(input, unsqrt);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});
        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }

    {
        auto input = std::make_shared<op::v0::Parameter>(element::f16, Shape{1, 2, 3});
        auto axes_const = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
        auto normalize_l2 = std::make_shared<op::v0::NormalizeL2>(input, axes_const, eps_value, op::EpsMode::MAX);

        model_ref = std::make_shared<ov::Model>(NodeVector{normalize_l2}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithMaxMulIncorrectSecondExp) {
    const float eps_value = 0.000099f;
    {
        auto input = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(3));
        auto exp = op::v0::Constant::create(element::f16, Shape{}, {2.f});
        auto pow = std::make_shared<op::v1::Power>(input, exp);
        auto axes_const = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<op::v1::ReduceSum>(pow, axes_const);
        auto eps_const = op::v0::Constant::create(element::f16, Shape{}, {eps_value});
        auto max = std::make_shared<op::v1::Maximum>(reduce_sum, eps_const);
        auto power_const = op::v0::Constant::create(element::f16, Shape{}, {-0.6f});
        auto unsqrt = std::make_shared<op::v1::Power>(max, power_const);
        auto mul = std::make_shared<op::v1::Multiply>(input, unsqrt);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});

        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithMaxSqrtAsPower) {
    const float eps_value = 0.000099f;
    {
        auto input = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(3));
        auto exp = op::v0::Constant::create(element::f16, Shape{}, {2.f});
        auto pow = std::make_shared<op::v1::Power>(input, exp);
        auto axes_const = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<op::v1::ReduceSum>(pow, axes_const);
        auto eps_const = op::v0::Constant::create(element::f16, Shape{}, {eps_value});
        auto max = std::make_shared<op::v1::Maximum>(reduce_sum, eps_const);
        auto sqrt_exp = op::v0::Constant::create(element::f16, Shape{}, {0.5f});
        auto sqrt = std::make_shared<op::v1::Power>(max, sqrt_exp);
        auto divide = std::make_shared<op::v1::Divide>(input, sqrt);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{input});
        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }

    {
        auto input = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(3));
        auto axes_const = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
        auto normalize_l2 = std::make_shared<op::v0::NormalizeL2>(input, axes_const, eps_value, op::EpsMode::MAX);

        model_ref = std::make_shared<ov::Model>(NodeVector{normalize_l2}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithMaxSqrtAsPowerIncorrectPowerExp) {
    const float eps_value = 0.0009f;
    {
        auto input = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(3));
        auto exp = op::v0::Constant::create(element::f16, Shape{}, {2.f});
        auto pow = std::make_shared<op::v1::Power>(input, exp);
        auto axes_const = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<op::v1::ReduceSum>(pow, axes_const);
        auto eps_const = op::v0::Constant::create(element::f16, Shape{}, {eps_value});
        auto max = std::make_shared<op::v1::Maximum>(reduce_sum, eps_const);
        auto sqrt_exp = op::v0::Constant::create(element::f16, Shape{}, {0.9f});
        auto sqrt = std::make_shared<op::v1::Power>(max, sqrt_exp);
        auto divide = std::make_shared<op::v1::Divide>(input, sqrt);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{input});

        manager.register_pass<ov::pass::NormalizeL2Fusion>();
    }
}
