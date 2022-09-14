// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/reduce_reshape_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ReduceMeanReshapeFusion) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {1});
    {
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto target_shape = opset9::Constant::create(element::i64, Shape{3}, {5, 1, 15});
        const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

        function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
        manager.register_pass<pass::ReduceReshapeFusion>();
    }
    {
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes, true);
        function_ref = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMeanReshapeFusion_ManyAxes) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15, 20});
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {1, 2});
    {
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {5, 1, 1, 20});
        const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

        function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
        manager.register_pass<pass::ReduceReshapeFusion>();
    }
    {
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes, true);
        function_ref = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMeanReshapeFusion_ManyAxes_2) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15, 1});
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {1, 2});
    {
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {5, 1, 1, 1});
        const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

        function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
        manager.register_pass<pass::ReduceReshapeFusion>();
    }
    {
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes, true);
        function_ref = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMeanReshapeFusion_SpecialZero) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15, 20});
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {2, 3});
    {
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {-1, 0, 1, 1});
        const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, true);

        function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
        manager.register_pass<pass::ReduceReshapeFusion>();
    }
    {
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes, true);
        function_ref = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMeanReshapeFusion_NegativeReshapeAxes) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15, 20});
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {-1, -3});
    {
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {5, 1, 15, 1});
        const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

        function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
        manager.register_pass<pass::ReduceReshapeFusion>();
    }
    {
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes, true);
        function_ref = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMeanReshapeFusion_NegativeReduceAxes) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15, 20});
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {-2, -3});
    {
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {5, 1, 1, 20});
        const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

        function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
        manager.register_pass<pass::ReduceReshapeFusion>();
    }
    {
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes, true);
        function_ref = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceOrReshapeFusion) {
    const auto input = std::make_shared<opset9::Parameter>(element::boolean, PartialShape{5, 10, 15, 20});
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {1, 2});
    {
        const auto reduce_mean = std::make_shared<opset9::ReduceLogicalOr>(input, reduce_axes);
        const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {5, 1, 1, 20});
        const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

        function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
        manager.register_pass<pass::ReduceReshapeFusion>();
    }
    {
        const auto reduce_mean = std::make_shared<opset9::ReduceLogicalOr>(input, reduce_axes, true);
        function_ref = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMeanReshapeFusion_SkipIfOneInNotAxisPosition) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15, 1});
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {1, 2});
    const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
    const auto target_shape = opset9::Constant::create(element::i64, Shape{5}, {5, 1, 1, 1, 1});
    const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

    function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    manager.register_pass<pass::ReduceReshapeFusion>();
}

TEST_F(TransformationTestsF, ReduceMeanReshapeFusion_SkipIfReshapeNotCompatible) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15, 20});
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {1, 2});
    const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
    const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {20, 1, 1, 5});
    const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

    function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    manager.register_pass<pass::ReduceReshapeFusion>();
}

TEST_F(TransformationTestsF, ReduceMeanReshapeFusion_SkipIfReshapeRankLessThanReduceRank) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {2});
    const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
    const auto target_shape = opset9::Constant::create(element::i64, Shape{1}, {50});
    const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

    function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    manager.register_pass<pass::ReduceReshapeFusion>();
}

TEST_F(TransformationTestsF, ReduceMeanReshapeFusion_SkipIfKeepDims) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {1});
    const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes, true);
    const auto target_shape = opset9::Constant::create(element::i64, Shape{3}, {5, 1, 15});
    const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

    function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    manager.register_pass<pass::ReduceReshapeFusion>();
}

TEST_F(TransformationTestsF, ReduceMeanReshapeFusion_SkipIfNonConstReduceAxes) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto reduce_axes = std::make_shared<opset9::Parameter>(element::i64, PartialShape{1});
    const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
    const auto target_shape = opset9::Constant::create(element::i64, Shape{3}, {5, 1, 15});
    const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

    function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input, reduce_axes});
    manager.register_pass<pass::ReduceReshapeFusion>();
}

TEST_F(TransformationTestsF, ReduceMeanReshapeFusion_SkipIfNonConstReshapeTargetShape) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {1});
    const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
    const auto target_shape = std::make_shared<opset9::Parameter>(element::i64, PartialShape{3});
    const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

    function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input, target_shape});
    manager.register_pass<pass::ReduceReshapeFusion>();
}
