// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/activations_scaling.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, ScaleDownSingleLayerTest) {
    float scale_factor = 128.f;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 16, 16});
        auto weights_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{3, 3, 3, 3}, {1});
        auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                              weights_const,
                                                              Strides{},
                                                              CoordinateDiff{},
                                                              CoordinateDiff{},
                                                              Strides{});
        auto convert = std::make_shared<ov::op::v0::Convert>(conv, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::activations_scaling::ScaleDownSingleLayer>(scale_factor);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 16, 16});
        auto weights_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{3, 3, 3, 3}, {1});
        auto scale_down_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {1.f / scale_factor});
        auto scale_down = std::make_shared<ov::op::v1::Multiply>(input, scale_down_const);
        auto conv = std::make_shared<ov::op::v1::Convolution>(scale_down,
                                                              weights_const,
                                                              Strides{},
                                                              CoordinateDiff{},
                                                              CoordinateDiff{},
                                                              Strides{});
        auto scale_up_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {scale_factor});
        auto scale_up = std::make_shared<ov::op::v1::Multiply>(conv, scale_up_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(scale_up, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, MulMulAddTransformationTest) {
    {
        auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 16, 16});
        auto scale_const_0 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul0 = std::make_shared<ov::op::v1::Multiply>(input0, scale_const_0);
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 16, 16});
        auto scale_const_1 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(input1, scale_const_1);
        auto add = std::make_shared<ov::op::v1::Add>(mul0, mul1);
        auto convert = std::make_shared<ov::op::v0::Convert>(add, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input0, input1});
        manager.register_pass<ov::pass::activations_scaling::MulMulAddTransformation>();
    }
    {
        auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 16, 16});
        auto scale_const_0 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul0 = std::make_shared<ov::op::v1::Multiply>(input0, scale_const_0);
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 16, 16});
        auto add = std::make_shared<ov::op::v1::Add>(mul0, input1);
        auto scale_const_1 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(add, scale_const_1);
        auto convert = std::make_shared<ov::op::v0::Convert>(mul1, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input0, input1});
    }
}

TEST_F(TransformationTestsF, MulGroupNormTransformationTest) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 16, 16});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul = std::make_shared<ov::op::v1::Multiply>(input, scale_const);
        auto norm_scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{3}, {10});
        auto norm_bias_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{3}, {10});
        auto group_norm =
            std::make_shared<ov::op::v12::GroupNormalization>(mul, norm_scale_const, norm_bias_const, 1, 0.01f);
        auto convert = std::make_shared<ov::op::v0::Convert>(group_norm, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::activations_scaling::MulGroupNormTransformation>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 16, 16});
        auto norm_scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{3}, {10});
        auto norm_bias_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{3}, {10});
        auto group_norm =
            std::make_shared<ov::op::v12::GroupNormalization>(input, norm_scale_const, norm_bias_const, 1, 0.01f);
        auto convert = std::make_shared<ov::op::v0::Convert>(group_norm, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, MulMVNTransformationTest) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 224, 224});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul = std::make_shared<ov::op::v1::Multiply>(input, scale_const);
        auto norm_axes_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, 2, 3});
        auto mvn =
            std::make_shared<ov::op::v6::MVN>(mul, norm_axes_const, true, 0.01f, ov::op::MVNEpsMode::INSIDE_SQRT);
        auto convert = std::make_shared<ov::op::v0::Convert>(mvn, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::activations_scaling::MulMVNTransformation>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 224, 224});
        auto norm_axes_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, 2, 3});
        auto mvn =
            std::make_shared<ov::op::v6::MVN>(input, norm_axes_const, true, 0.01f, ov::op::MVNEpsMode::INSIDE_SQRT);
        auto convert = std::make_shared<ov::op::v0::Convert>(mvn, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SplitTransformationTest) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul = std::make_shared<ov::op::v1::Multiply>(input, scale_const);
        auto axis = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {0});
        auto split_length = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, 2, 3});
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(mul, axis, split_length);
        auto convert0 = std::make_shared<ov::op::v0::Convert>(split->output(0), ov::element::f32);
        auto result0 = std::make_shared<ov::op::v0::Result>(convert0);
        auto convert1 = std::make_shared<ov::op::v0::Convert>(split->output(1), ov::element::f32);
        auto result1 = std::make_shared<ov::op::v0::Result>(convert1);
        auto convert2 = std::make_shared<ov::op::v0::Convert>(split->output(2), ov::element::f32);
        auto result2 = std::make_shared<ov::op::v0::Result>(convert2);

        model = std::make_shared<ov::Model>(ov::ResultVector{result0, result1, result2}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::activations_scaling::SplitTransformation>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto axis = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {0});
        auto split_length = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, 2, 3});
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis, split_length);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul0 = std::make_shared<ov::op::v1::Multiply>(split->output(0), scale_const);
        auto convert0 = std::make_shared<ov::op::v0::Convert>(mul0, ov::element::f32);
        auto result0 = std::make_shared<ov::op::v0::Result>(convert0);
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(split->output(1), scale_const);
        auto convert1 = std::make_shared<ov::op::v0::Convert>(mul1, ov::element::f32);
        auto result1 = std::make_shared<ov::op::v0::Result>(convert1);
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(split->output(2), scale_const);
        auto convert2 = std::make_shared<ov::op::v0::Convert>(mul2, ov::element::f32);
        auto result2 = std::make_shared<ov::op::v0::Result>(convert2);

        model_ref =
            std::make_shared<ov::Model>(ov::ResultVector{result0, result1, result2}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ReshapeTransformationTest) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul = std::make_shared<ov::op::v1::Multiply>(input, scale_const);
        auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 0, 1, -1});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(mul, shape, true);
        auto convert = std::make_shared<ov::op::v0::Convert>(reshape, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::activations_scaling::ReshapeTransformation>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 0, 1, -1});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(input, shape, true);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul = std::make_shared<ov::op::v1::Multiply>(reshape, scale_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(mul, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, MulMulMulTransformationTest) {
    {
        auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto scale_const0 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul0 = std::make_shared<ov::op::v1::Multiply>(input0, scale_const0);
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto scale_const1 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(input1, scale_const1);
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(mul0, mul1);
        auto convert = std::make_shared<ov::op::v0::Convert>(mul2, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input0, input1});
        manager.register_pass<ov::pass::activations_scaling::MulMulMulTransformation>();
    }
    {
        auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto mul = std::make_shared<ov::op::v1::Multiply>(input0, input1);
        auto new_scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto new_mul = std::make_shared<ov::op::v1::Multiply>(mul, new_scale_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(new_mul, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input0, input1});
    }
}

TEST_F(TransformationTestsF, ConcatTransformationTest) {
    {
        auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto scale_const0 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul0 = std::make_shared<ov::op::v1::Multiply>(input0, scale_const0);
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto scale_const1 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(input1, scale_const1);
        auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector{mul0, mul1}, 0);
        auto convert = std::make_shared<ov::op::v0::Convert>(concat, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input0, input1});
        manager.register_pass<ov::pass::activations_scaling::ConcatTransformation>();
    }
    {
        auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto scale_const0 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul0 = std::make_shared<ov::op::v1::Multiply>(input0, scale_const0);
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto scale_const1 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(input1, scale_const1);
        auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector{mul0, mul1}, 0);
        auto new_scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto new_mul = std::make_shared<ov::op::v1::Multiply>(concat, new_scale_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(new_mul, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input0, input1});
    }
}
