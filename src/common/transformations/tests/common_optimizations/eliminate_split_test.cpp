// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <queue>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/opsets/opset8.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"

using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, EliminateSplit) {
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic());
        auto mul_constant = opset8::Constant::create(element::f32, Shape{1}, {89.2});
        auto mul = std::make_shared<opset8::Multiply>(input, mul_constant);
        auto axis_const = opset8::Constant::create(element::i64, Shape{}, {2});
        auto split = std::make_shared<opset8::Split>(mul, axis_const, 1);
        auto res = std::make_shared<opset8::Result>(split);
        model = std::make_shared<ov::Model>(NodeVector{res}, ParameterVector{input});

        manager.register_pass<ov::pass::EliminateSplit>();
    }
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic());
        auto mul_constant = opset8::Constant::create(element::f32, Shape{1}, {89.2});
        auto mul = std::make_shared<opset8::Multiply>(input, mul_constant);
        auto res = std::make_shared<opset8::Result>(mul);
        model_ref = std::make_shared<ov::Model>(NodeVector{res}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, EliminateSplitNegative) {
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic());
        auto mul_constant = opset8::Constant::create(element::f32, Shape{1}, {89.2});
        auto mul = std::make_shared<opset8::Multiply>(input, mul_constant);
        auto axis_const = opset8::Constant::create(element::i64, Shape{}, {2});
        auto split = std::make_shared<opset8::Split>(mul, axis_const, 3);
        auto res1 = std::make_shared<opset8::Result>(split->output(0));
        auto res2 = std::make_shared<opset8::Result>(split->output(1));
        auto res3 = std::make_shared<opset8::Result>(split->output(2));
        model = std::make_shared<ov::Model>(NodeVector{res1, res2, res3}, ParameterVector{input});

        manager.register_pass<ov::pass::EliminateSplit>();
    }
}

TEST_F(TransformationTestsF, EliminateSequenceOfSplits) {
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic());
        auto axis_const1 = opset8::Constant::create(element::i64, Shape{}, {0});
        auto axis_const2 = opset8::Constant::create(element::i64, Shape{}, {1});
        auto axis_const3 = opset8::Constant::create(element::i64, Shape{}, {2});
        auto split1 = std::make_shared<opset8::Split>(input, axis_const1, 1);
        auto split2 = std::make_shared<opset8::Split>(split1, axis_const2, 1);
        auto split3 = std::make_shared<opset8::Split>(split2, axis_const3, 1);
        auto axis_const = opset8::Constant::create(element::i64, Shape{}, {2});
        auto true_split = std::make_shared<opset8::Split>(split3, axis_const, 3);
        auto res1 = std::make_shared<opset8::Result>(true_split->output(0));
        auto res2 = std::make_shared<opset8::Result>(true_split->output(1));
        auto res3 = std::make_shared<opset8::Result>(true_split->output(2));
        model = std::make_shared<ov::Model>(NodeVector{res1, res2, res3}, ParameterVector{input});

        manager.register_pass<ov::pass::EliminateSplit>();
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic());
        auto axis_const = opset8::Constant::create(element::i64, Shape{}, {2});
        auto split = std::make_shared<opset8::Split>(input, axis_const, 3);
        auto res1 = std::make_shared<opset8::Result>(split->output(0));
        auto res2 = std::make_shared<opset8::Result>(split->output(1));
        auto res3 = std::make_shared<opset8::Result>(split->output(2));
        model_ref = std::make_shared<ov::Model>(NodeVector{res1, res2, res3}, ParameterVector{input});
    }
}
