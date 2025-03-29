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
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
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
        auto bias_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 3, 1, 1}, {2.3f});
        auto add = std::make_shared<ov::op::v1::Add>(conv, bias_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(add, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::activations_scaling::ScaleDownSingleLayer>(scale_factor, ov::element::f16);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 16, 16});
        auto weights_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{3, 3, 3, 3}, {1});
        auto scale_down_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {1.f / scale_factor});
        auto scale_down = std::make_shared<ov::op::v1::Multiply>(input, scale_down_const);
        auto conv = std::make_shared<ov::op::v1::Convolution>(scale_down,
                                                              weights_const,
                                                              Strides{},
                                                              CoordinateDiff{},
                                                              CoordinateDiff{},
                                                              Strides{});
        auto bias_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 3, 1, 1}, {2.3f});
        auto scale_down_bias = std::make_shared<ov::op::v1::Multiply>(bias_const, scale_down_const);
        auto add = std::make_shared<ov::op::v1::Add>(conv, scale_down_bias);
        auto scale_up_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {scale_factor});
        auto scale_up = std::make_shared<ov::op::v1::Multiply>(add, scale_up_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(scale_up, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, EliminateScalarMulTest) {
    double epsilon = 1.f;
    float scale_factor = 8.f;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 4, 4});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {scale_factor});
        auto mul = std::make_shared<ov::op::v1::Multiply>(input, scale_const);
        auto norm_scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{3}, {10});
        auto norm_bias_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{3}, {10});
        auto group_norm =
            std::make_shared<ov::op::v12::GroupNormalization>(mul, norm_scale_const, norm_bias_const, 1, epsilon);
        auto convert = std::make_shared<ov::op::v0::Convert>(group_norm, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::activations_scaling::EliminateScalarMul>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 4, 4});
        auto norm_scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{3}, {10});
        auto norm_bias_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{3}, {10});
        epsilon /= scale_factor;
        auto group_norm =
            std::make_shared<ov::op::v12::GroupNormalization>(input, norm_scale_const, norm_bias_const, 1, epsilon);
        auto convert = std::make_shared<ov::op::v0::Convert>(group_norm, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, MoveDownScalarMulTest) {
    {
        auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto scale_const0 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul0 = std::make_shared<ov::op::v1::Multiply>(input0, scale_const0);
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(input1, mul0);
        auto convert = std::make_shared<ov::op::v0::Convert>(mul1, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input0, input1});
        manager.register_pass<ov::pass::activations_scaling::MoveDownScalarMul>();
    }
    {
        auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{6, 12, 10, 24});
        auto mul0 = std::make_shared<ov::op::v1::Multiply>(input0, input1);
        auto scale_const0 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {10});
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(mul0, scale_const0);
        auto convert = std::make_shared<ov::op::v0::Convert>(mul1, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input0, input1});
    }
}

TEST_F(TransformationTestsF, MulShareTransformationTest) {
    float epsilon = 1.f;
    float scale_factor = 8.f;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 4, 4});
        auto mvn_axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 3});
        auto mvn = std::make_shared<ov::op::v6::MVN>(input, mvn_axes, true, epsilon, ov::op::MVNEpsMode::INSIDE_SQRT);
        auto convert0 = std::make_shared<ov::op::v0::Convert>(mvn, ov::element::f32);
        auto result0 = std::make_shared<ov::op::v0::Result>(convert0);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {scale_factor});
        auto mul = std::make_shared<ov::op::v1::Multiply>(input, scale_const);
        auto convert1 = std::make_shared<ov::op::v0::Convert>(mul, ov::element::f32);
        auto result1 = std::make_shared<ov::op::v0::Result>(convert1);

        model = std::make_shared<ov::Model>(ov::ResultVector{result0, result1}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::activations_scaling::MulShareTransformation>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 4, 4});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {scale_factor});
        auto mul = std::make_shared<ov::op::v1::Multiply>(input, scale_const);
        auto mvn_axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 3});
        epsilon *= scale_factor * scale_factor;
        auto mvn = std::make_shared<ov::op::v6::MVN>(mul, mvn_axes, true, epsilon, ov::op::MVNEpsMode::INSIDE_SQRT);
        auto convert0 = std::make_shared<ov::op::v0::Convert>(mvn, ov::element::f32);
        auto result0 = std::make_shared<ov::op::v0::Result>(convert0);
        auto convert1 = std::make_shared<ov::op::v0::Convert>(mul, ov::element::f32);
        auto result1 = std::make_shared<ov::op::v0::Result>(convert1);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result0, result1}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
