// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/adaptive_pool_to_reduce.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, AdaptiveAvgPool2dToReduceMean) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 14, 14});
        auto out_spatial_shape = opset10::Constant::create(element::i32, Shape{2}, {1, 1});
        auto adaptive_pool = std::make_shared<opset10::AdaptiveAvgPool>(data, out_spatial_shape);
        auto result = std::make_shared<opset10::Result>(adaptive_pool);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
        manager.register_pass<pass::AdaptivePoolToReduce>();
    }
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 14, 14});
        auto axes = opset10::Constant::create(element::i64, Shape{2}, {2, 3});
        auto reduce_mean = std::make_shared<opset10::ReduceMean>(data, axes, true);
        auto result = std::make_shared<opset10::Result>(reduce_mean);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, AdaptiveMaxPool2dToReduceMax) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 14, 14});
        auto out_spatial_shape = opset10::Constant::create(element::i32, Shape{2}, {1, 1});
        auto adaptive_pool = std::make_shared<opset10::AdaptiveMaxPool>(data, out_spatial_shape);
        auto result = std::make_shared<opset10::Result>(adaptive_pool);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
        manager.register_pass<pass::AdaptivePoolToReduce>();
    }
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 14, 14});
        auto axes = opset10::Constant::create(element::i64, Shape{2}, {2, 3});
        auto reduce_mean = std::make_shared<opset10::ReduceMax>(data, axes, true);
        auto result = std::make_shared<opset10::Result>(reduce_mean);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, AdaptiveMaxPool2dToReduceMaxUsedIndexes) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 14, 14});
        auto out_spatial_shape = opset10::Constant::create(element::i32, Shape{2}, {1, 1});
        auto adaptive_pool = std::make_shared<opset10::AdaptiveMaxPool>(data, out_spatial_shape);
        auto result1 = std::make_shared<opset10::Result>(adaptive_pool->output(0));
        auto result2 = std::make_shared<opset10::Result>(adaptive_pool->output(1));
        model = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{data});
        manager.register_pass<pass::AdaptivePoolToReduce>();
    }
    // Reference model equals initial model
}

TEST_F(TransformationTestsF, AdaptiveAvgPool3dToReduceMean) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 14, 14, 14});
        auto out_spatial_shape = opset10::Constant::create(element::i32, Shape{3}, {1, 1, 1});
        auto adaptive_pool = std::make_shared<opset10::AdaptiveAvgPool>(data, out_spatial_shape);
        auto result = std::make_shared<opset10::Result>(adaptive_pool);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
        manager.register_pass<pass::AdaptivePoolToReduce>();
    }
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 14, 14, 14});
        auto axes = opset10::Constant::create(element::i64, Shape{3}, {2, 3, 4});
        auto reduce_mean = std::make_shared<opset10::ReduceMean>(data, axes, true);
        auto result = std::make_shared<opset10::Result>(reduce_mean);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, AdaptiveMaxPool3dToReduceMax) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 14, 14, 14});
        auto out_spatial_shape = opset10::Constant::create(element::i32, Shape{3}, {1, 1, 1});
        auto adaptive_pool = std::make_shared<opset10::AdaptiveMaxPool>(data, out_spatial_shape);
        auto result = std::make_shared<opset10::Result>(adaptive_pool);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
        manager.register_pass<pass::AdaptivePoolToReduce>();
    }
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 14, 14, 14});
        auto axes = opset10::Constant::create(element::i64, Shape{3}, {2, 3, 4});
        auto reduce_mean = std::make_shared<opset10::ReduceMax>(data, axes, true);
        auto result = std::make_shared<opset10::Result>(reduce_mean);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    }
}
