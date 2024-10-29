// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/activations_scaling.hpp"

#include <gtest/gtest.h>


#include <string>

#include <memory>
#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
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
        manager.register_pass<ov::pass::ScaleDownSingleLayer>(scale_factor);
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

TEST_F(TransformationTestsF, MulMulAddFusionTest) {
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
        manager.register_pass<ov::pass::MulMulAddFusion>();
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

TEST_F(TransformationTestsF, MulGroupNormFusionTest) {
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
        manager.register_pass<ov::pass::MulGroupNormFusion>();
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
