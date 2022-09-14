// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/pull_through_reduce.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, PullUnsqueezeThroughReduceMean) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto unsqueeze_axes = opset9::Constant::create(element::i64, Shape{}, {0});
    {
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(input, unsqueeze_axes);
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {2});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(unsqueeze, reduce_axes);

        function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullUnsqueezeThroughReduce>();
    }
    {
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {1});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(reduce_mean, unsqueeze_axes);

        function_ref = std::make_shared<Model>(NodeVector{unsqueeze}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullUnsqueezeThroughReduceLogicalOr) {
    const auto input = std::make_shared<opset9::Parameter>(element::boolean, PartialShape{5, 10, 15});
    const auto unsqueeze_axes = opset9::Constant::create(element::i64, Shape{}, {0});
    {
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(input, unsqueeze_axes);
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {2});
        const auto reduce_or = std::make_shared<opset9::ReduceLogicalOr>(unsqueeze, reduce_axes);

        function = std::make_shared<Model>(NodeVector{reduce_or}, ParameterVector{input});
        manager.register_pass<pass::PullUnsqueezeThroughReduce>();
    }
    {
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {1});
        const auto reduce_or = std::make_shared<opset9::ReduceLogicalOr>(input, reduce_axes);
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(reduce_or, unsqueeze_axes);

        function_ref = std::make_shared<Model>(NodeVector{unsqueeze}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, PullUnsqueezeThroughReduceMean_UnsqueezeAxesGreaterThanReduceAxes) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15, 20});
    const auto unsqueeze_axes = opset9::Constant::create(element::i64, Shape{3}, {0, 2, 3});
    {
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(input, unsqueeze_axes);
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {5, 6});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(unsqueeze, reduce_axes);

        function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullUnsqueezeThroughReduce>();
    }
    {
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {2, 3});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(reduce_mean, unsqueeze_axes);

        function_ref = std::make_shared<Model>(NodeVector{unsqueeze}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullUnsqueezeThroughReduceMean_UnsqueezeAxesLowerThanReduceAxes) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {0, 2});
    {
        const auto unsqueeze_axes = opset9::Constant::create(element::i64, Shape{3}, {3, 4, 5});
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(input, unsqueeze_axes);
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(unsqueeze, reduce_axes);

        function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullUnsqueezeThroughReduce>();
    }
    {
        const auto unsqueeze_axes = opset9::Constant::create(element::i64, Shape{3}, {1, 2, 3});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(reduce_mean, unsqueeze_axes);

        function_ref = std::make_shared<Model>(NodeVector{unsqueeze}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullUnsqueezeThroughReduceMean_UnsqueezeAxesBetweenReduceAxes) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    {
        const auto unsqueeze_axes = opset9::Constant::create(element::i64, Shape{3}, {1, 3, 5});
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{3}, {0, 2, 4});
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(input, unsqueeze_axes);
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(unsqueeze, reduce_axes);

        function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullUnsqueezeThroughReduce>();
    }
    {
        const auto unsqueeze_axes = opset9::Constant::create(element::i64, Shape{3}, {0, 1, 2});
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{3}, {0, 1, 2});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(reduce_mean, unsqueeze_axes);

        function_ref = std::make_shared<Model>(NodeVector{unsqueeze}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullUnsqueezeThroughReduceMean_UnsqueezeAxesBetweenReduceAxes_2) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{1, 10, 1, 20});
    {
        const auto unsqueeze_axes = opset9::Constant::create(element::i64, Shape{4}, {0, 1, 3, 4});
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {2, 5});
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(input, unsqueeze_axes);
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(unsqueeze, reduce_axes);

        function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullUnsqueezeThroughReduce>();
    }
    {
        const auto unsqueeze_axes = opset9::Constant::create(element::i64, Shape{4}, {0, 1, 2, 3});
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {0, 1});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(reduce_mean, unsqueeze_axes);

        function_ref = std::make_shared<Model>(NodeVector{unsqueeze}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullUnsqueezeThroughReduceMean_UnsqueezeAxesBetweenReduceAxes_KeepDims) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto unsqueeze_axes = opset9::Constant::create(element::i64, Shape{3}, {1, 3, 5});
    {
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{3}, {0, 2, 4});
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(input, unsqueeze_axes);
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(unsqueeze, reduce_axes, true);

        function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullUnsqueezeThroughReduce>();
    }
    {
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{3}, {0, 1, 2});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes, true);
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(reduce_mean, unsqueeze_axes);

        function_ref = std::make_shared<Model>(NodeVector{unsqueeze}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullUnsqueezeThroughReduceLogicalOr_UnsqueezeAxesBetweenReduceAxes_KeepDims) {
    const auto input = std::make_shared<opset9::Parameter>(element::boolean, PartialShape{1, 10, 1, 20});
    const auto unsqueeze_axes = opset9::Constant::create(element::i64, Shape{4}, {0, 1, 3, 4});
    {
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {2, 5});
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(input, unsqueeze_axes);
        const auto reduce_or = std::make_shared<opset9::ReduceLogicalOr>(unsqueeze, reduce_axes, true);

        function = std::make_shared<Model>(NodeVector{reduce_or}, ParameterVector{input});
        manager.register_pass<pass::PullUnsqueezeThroughReduce>();
    }
    {
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {0, 1});
        const auto reduce_or = std::make_shared<opset9::ReduceLogicalOr>(input, reduce_axes, true);
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(reduce_or, unsqueeze_axes);

        function_ref = std::make_shared<Model>(NodeVector{unsqueeze}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, PullUnsqueezeThroughReduceMean_UnsqueezeAxesBetweenReduceAxes_NegativeAxes) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    {
        const auto unsqueeze_axes = opset9::Constant::create(element::i64, Shape{3}, {1, -3, -1});
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{3}, {-2, 2, 0});
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(input, unsqueeze_axes);
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(unsqueeze, reduce_axes);

        function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullUnsqueezeThroughReduce>();
    }
    {
        const auto unsqueeze_axes = opset9::Constant::create(element::i64, Shape{3}, {0, 1, 2});
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{3}, {0, 1, 2});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(reduce_mean, unsqueeze_axes);

        function_ref = std::make_shared<Model>(NodeVector{unsqueeze}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullUnsqueezeThroughReduceMean_DynamicInput) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, Dimension::dynamic(), 15});
    const auto unsqueeze_axes = opset9::Constant::create(element::i64, Shape{}, {0});
    {
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(input, unsqueeze_axes);
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {2});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(unsqueeze, reduce_axes);

        function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullUnsqueezeThroughReduce>();
    }
    {
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {1});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(reduce_mean, unsqueeze_axes);

        function_ref = std::make_shared<Model>(NodeVector{unsqueeze}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullUnsqueezeThroughReduce_SkipIfTheSameAxes) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto unsqueeze_axes = opset9::Constant::create(element::i64, Shape{2}, {0, 1});
    const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(input, unsqueeze_axes);
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {1, 2});
    const auto reduce_mean = std::make_shared<opset9::ReduceMean>(unsqueeze, reduce_axes);

    function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
    manager.register_pass<pass::PullUnsqueezeThroughReduce>();
}

TEST_F(TransformationTestsF, PullUnsqueezeThroughReduce_SkipIfNotConstAxes) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, Dimension::dynamic(), 15});
    const auto unsqueeze_axes = std::make_shared<opset9::Parameter>(element::i64, Shape{});
    const auto unsqueeze = std::make_shared<opset9::Unsqueeze>(input, unsqueeze_axes);
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {2});
    const auto reduce_mean = std::make_shared<opset9::ReduceMean>(unsqueeze, reduce_axes);

    function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input, unsqueeze_axes});
    manager.register_pass<pass::PullUnsqueezeThroughReduce>();
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMean) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    {
        const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {1, 5, 10, 15});
        const auto reshape = std::make_shared<opset9::Reshape>(input, target_shape, false);
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {2});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(reshape, reduce_axes);

        function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto target_shape = opset9::Constant::create(element::i64, Shape{3}, {1, 5, 15});
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {1});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

        function_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceLogicalOr) {
    const auto input = std::make_shared<opset9::Parameter>(element::boolean, PartialShape{5, 10, 15});
    {
        const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {1, 5, 10, 15});
        const auto reshape = std::make_shared<opset9::Reshape>(input, target_shape, false);
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {2});
        const auto reduce_or = std::make_shared<opset9::ReduceLogicalOr>(reshape, reduce_axes);

        function = std::make_shared<Model>(NodeVector{reduce_or}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {1});
        const auto reduce_or = std::make_shared<opset9::ReduceLogicalOr>(input, reduce_axes);
        const auto target_shape = opset9::Constant::create(element::i64, Shape{3}, {1, 5, 15});
        const auto reshape = std::make_shared<opset9::Reshape>(reduce_or, target_shape, false);

        function_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMean_InsertAxesAtTheEnd) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {1, 2});
    {
        const auto target_shape = opset9::Constant::create(element::i64, Shape{5}, {5, 10, 15, 1, 1});
        const auto reshape = std::make_shared<opset9::Reshape>(input, target_shape, false);
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(reshape, reduce_axes);

        function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto target_shape = opset9::Constant::create(element::i64, Shape{3}, {5, 1, 1});
        const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

        function_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMean_InsertAxesAtTheBegin) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    {
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {3, 5});
        const auto target_shape = opset9::Constant::create(element::i64, Shape{6}, {1, 1, 1, 5, 10, 15});
        const auto reshape = std::make_shared<opset9::Reshape>(input, target_shape, false);
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(reshape, reduce_axes);

        function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {0, 2});
        const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {1, 1, 1, 10});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

        function_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMean_InsertAxesOnBothSides) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    {
        const auto target_shape = opset9::Constant::create(element::i64, Shape{6}, {1, 1, 5, 10, 15, 1});
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {2, 3});
        const auto reshape = std::make_shared<opset9::Reshape>(input, target_shape, false);
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(reshape, reduce_axes);

        function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {0, 1});
        const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {1, 1, 15, 1});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

        function_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMean_InsertAxesOnBothSides_2) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{1, 5, 10, 1});
    {
        const auto target_shape = opset9::Constant::create(element::i64, Shape{6}, {1, 1, 5, 10, 1, 1});
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {2, 3});
        const auto reshape = std::make_shared<opset9::Reshape>(input, target_shape, false);
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(reshape, reduce_axes);

        function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {1, 2});
        const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

        function_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMean_KeepDims) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    {
        const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {1, 5, 10, 15});
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{3}, {1, 2, 3});
        const auto reshape = std::make_shared<opset9::Reshape>(input, target_shape, false);
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(reshape, reduce_axes, true);

        function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1});
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{3}, {0, 1, 2});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes, true);
        const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

        function_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceLogicalOr_KeepDims) {
    const auto input = std::make_shared<opset9::Parameter>(element::boolean, PartialShape{1, 10, 1, 20});
    {
        const auto target_shape = opset9::Constant::create(element::i64, Shape{5}, {1, 1, 10, 1, 20});
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {2, 4});
        const auto reshape = std::make_shared<opset9::Reshape>(input, target_shape, false);
        const auto reduce_or = std::make_shared<opset9::ReduceLogicalOr>(reshape, reduce_axes, true);

        function = std::make_shared<Model>(NodeVector{reduce_or}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto target_shape = opset9::Constant::create(element::i64, Shape{5}, {1, 1, 1, 1, 1});
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {1, 3});
        const auto reduce_or = std::make_shared<opset9::ReduceLogicalOr>(input, reduce_axes, true);
        const auto reshape = std::make_shared<opset9::Reshape>(reduce_or, target_shape, false);

        function_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMean_NegativeAxes) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    {
        const auto target_shape = opset9::Constant::create(element::i64, Shape{5}, {1, 5, 10, 15, 1});
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {-2, -3});
        const auto reshape = std::make_shared<opset9::Reshape>(input, target_shape, false);
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(reshape, reduce_axes);

        function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto target_shape = opset9::Constant::create(element::i64, Shape{3}, {1, 5, 1});
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{2}, {1, 2});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, false);

        function_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduce_SpecialZeroTrue) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    {
        const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {5, 0, -1, 1});
        const auto reshape = std::make_shared<opset9::Reshape>(input, target_shape, true);
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {2});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(reshape, reduce_axes);

        function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto target_shape = opset9::Constant::create(element::i64, Shape{3}, {5, 10, 1});
        const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {2});
        const auto reduce_mean = std::make_shared<opset9::ReduceMean>(input, reduce_axes);
        const auto reshape = std::make_shared<opset9::Reshape>(reduce_mean, target_shape, true);

        function_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMean_SkipIfDynamicInput) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, Dimension::dynamic(), 15});
    const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {1, 5, 10, 15});
    const auto reshape = std::make_shared<opset9::Reshape>(input, target_shape, false);
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {2});
    const auto reduce_mean = std::make_shared<opset9::ReduceMean>(reshape, reduce_axes);

    function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
    manager.register_pass<pass::PullReshapeThroughReduce>();
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMean_SkipIfDynamicReshapeOutputShape) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto target_shape = std::make_shared<opset9::Parameter>(element::i32, PartialShape{4});
    const auto reshape = std::make_shared<opset9::Reshape>(input, target_shape, false);
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {2});
    const auto reduce_mean = std::make_shared<opset9::ReduceMean>(reshape, reduce_axes);

    function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input, target_shape});
    manager.register_pass<pass::PullReshapeThroughReduce>();
}

TEST_F(TransformationTestsF, PullReshapeThroughReduce_SkipIfNotConstAxes) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {1, 5, 10, 15});
    const auto reshape = std::make_shared<opset9::Reshape>(input, target_shape, false);
    const auto reduce_axes = std::make_shared<opset9::Parameter>(element::i64, PartialShape{});
    const auto reduce_mean = std::make_shared<opset9::ReduceMean>(reshape, reduce_axes);

    function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input, reduce_axes});
    manager.register_pass<pass::PullReshapeThroughReduce>();
}

TEST_F(TransformationTestsF, PullReshapeThroughReduce_SkipIfTheSameAxes) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {1, 5, 10, 15});
    const auto reshape = std::make_shared<opset9::Reshape>(input, target_shape, false);
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {0});
    const auto reduce_mean = std::make_shared<opset9::ReduceMean>(reshape, reduce_axes);

    function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
    manager.register_pass<pass::PullReshapeThroughReduce>();
}

TEST_F(TransformationTestsF, PullReshapeThroughReduce_SkipIfInsertAxesInTheMiddle) {
    const auto input = std::make_shared<opset9::Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto target_shape = opset9::Constant::create(element::i64, Shape{4}, {5, 10, 1, 15});
    const auto reshape = std::make_shared<opset9::Reshape>(input, target_shape, false);
    const auto reduce_axes = opset9::Constant::create(element::i64, Shape{}, {0});
    const auto reduce_mean = std::make_shared<opset9::ReduceMean>(reshape, reduce_axes);

    function = std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
    manager.register_pass<pass::PullReshapeThroughReduce>();
}
