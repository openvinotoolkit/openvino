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
        auto scale_up_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {scale_factor});
        auto scale_up = std::make_shared<ov::op::v1::Multiply>(conv, scale_up_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(scale_up, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ScaleDownFusionTest) {
    float scale_factor = 128.f;
    {
        ov::Shape scale_const_shape = {};
        std::vector<float> scale_down_value = {1.f / scale_factor};
        std::shared_ptr<ov::Node> scale_down_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f16, scale_const_shape, scale_down_value);

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 16, 16});
        auto shape_pre = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, 3, 256});
        auto reshape_pre = std::make_shared<ov::op::v1::Reshape>(input, shape_pre, true);
        auto shape_post = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 3, 16, 16});

        auto scale_down0 = std::make_shared<ov::op::v1::Multiply>(reshape_pre->output(0), scale_down_const);
        ov::pass::activations_scaling::mark_as_scale_down_node(scale_down0);
        auto reshape_post0 = std::make_shared<ov::op::v1::Reshape>(scale_down0, shape_post, true);
        auto result0 = std::make_shared<ov::op::v0::Result>(reshape_post0);

        auto scale_down1 = std::make_shared<ov::op::v1::Multiply>(reshape_pre->output(0), scale_down_const);
        ov::pass::activations_scaling::mark_as_scale_down_node(scale_down1);
        auto reshape_post1 = std::make_shared<ov::op::v1::Reshape>(scale_down1, shape_post, true);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape_post1);

        model = std::make_shared<ov::Model>(ov::ResultVector{result0, result1}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::activations_scaling::ScaleDownFusion>();
    }
    {
        ov::Shape scale_const_shape = {};
        std::vector<float> scale_down_value = {1.f / scale_factor};
        std::shared_ptr<ov::Node> scale_down_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f16, scale_const_shape, scale_down_value);

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 16, 16});
        auto shape_pre = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, 3, 256});
        auto reshape_pre = std::make_shared<ov::op::v1::Reshape>(input, shape_pre, true);
        auto shape_post = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 3, 16, 16});

        auto scale_down0 = std::make_shared<ov::op::v1::Multiply>(reshape_pre->output(0), scale_down_const);
        ov::pass::activations_scaling::mark_as_scale_down_node(scale_down0);
        auto reshape_post0 = std::make_shared<ov::op::v1::Reshape>(scale_down0, shape_post, true);
        auto result0 = std::make_shared<ov::op::v0::Result>(reshape_post0);

        auto reshape_post1 = std::make_shared<ov::op::v1::Reshape>(scale_down0, shape_post, true);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape_post1);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result0, result1}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, MulNormTransformationTest) {
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
        manager.register_pass<ov::pass::activations_scaling::MulNormTransformation>();
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
        manager.register_pass<ov::pass::activations_scaling::MulConcatTransformation>();
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
