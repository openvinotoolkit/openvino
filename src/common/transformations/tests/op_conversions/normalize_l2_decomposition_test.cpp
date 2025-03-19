// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/normalize_l2_decomposition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, NormalizeL2DecomositionFusionWithMax) {
    const float eps_value = 0.000099f;
    {
        auto input = std::make_shared<opset8::Parameter>(element::f16, PartialShape::dynamic(3));
        auto axes_const = opset8::Constant::create(element::i64, Shape{2}, {1, 2});
        auto normalize_l2 = std::make_shared<opset8::NormalizeL2>(input, axes_const, eps_value, op::EpsMode::MAX);

        model = std::make_shared<ov::Model>(NodeVector{normalize_l2}, ParameterVector{input});

        manager.register_pass<ov::pass::NormalizeL2Decomposition>();
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::f16, PartialShape::dynamic(3));
        auto exp = opset8::Constant::create(element::f16, Shape{}, {2.f});
        auto pow = std::make_shared<opset8::Power>(input, exp);
        auto axes_const = opset8::Constant::create(element::i64, Shape{2}, {1, 2});
        auto reduce_sum = std::make_shared<opset8::ReduceSum>(pow, axes_const, true);
        auto eps_const = opset8::Constant::create(element::f16, Shape{}, {eps_value});
        auto max = std::make_shared<opset8::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset8::Sqrt>(max);
        auto divide = std::make_shared<opset8::Divide>(input, sqrt);

        model_ref = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, NormalizeL2DecomositionFusionWithAdd) {
    const float eps_value = 0.000099f;
    {
        auto input = std::make_shared<opset8::Parameter>(element::f16, PartialShape::dynamic(3));
        auto axes_const = opset8::Constant::create(element::i64, Shape{2}, {0, 1});
        auto normalize_l2 = std::make_shared<opset8::NormalizeL2>(input, axes_const, eps_value, op::EpsMode::ADD);

        model = std::make_shared<ov::Model>(NodeVector{normalize_l2}, ParameterVector{input});

        manager.register_pass<ov::pass::NormalizeL2Decomposition>();
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::f16, PartialShape::dynamic(3));
        auto exp = opset8::Constant::create(element::f16, Shape{}, {2.f});
        auto pow = std::make_shared<opset8::Power>(input, exp);
        auto axes_const = opset8::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset8::ReduceSum>(pow, axes_const, true);
        auto eps_const = opset8::Constant::create(element::f16, Shape{}, {eps_value});
        auto max = std::make_shared<opset8::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset8::Sqrt>(max);
        auto divide = std::make_shared<opset8::Divide>(input, sqrt);

        model_ref = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{input});
    }
}
