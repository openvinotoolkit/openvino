// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ngraph_test_utils.hpp"

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/reduce_merge.hpp>

using namespace ngraph;

TEST_F(TransformationTestsF, ReduceMergeReduceL1) {
    {
        auto data = std::make_shared<op::Parameter>(element::f32, Shape{3, 2});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(data, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceL1>(reduce1, reduce2_axis, true)},
                                       ParameterVector{data});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::f32, Shape{3, 2});
        auto axes = op::Constant::create(element::i64, Shape{2}, {0, 0});
        auto reduce = std::make_shared<opset9::ReduceL1>(data, axes, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeReduceL2) {
    {
        auto data = std::make_shared<op::Parameter>(element::f32, Shape{3, 2});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(data, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceL2>(reduce1, reduce2_axis, true)},
                                       ParameterVector{data});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::f32, Shape{3, 2});
        auto axes = op::Constant::create(element::i64, Shape{2}, {0, 0});
        auto reduce = std::make_shared<opset9::ReduceL2>(data, axes, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeReduceLogicalAnd) {
    {
        auto data = std::make_shared<op::Parameter>(element::boolean, Shape{3, 2});
        auto reduce1_axis = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceLogicalAnd>(data, reduce1_axis, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function = std::make_shared<Function>(
            OutputVector{std::make_shared<opset9::ReduceLogicalAnd>(reduce1, reduce2_axis, true)},
            ParameterVector{data});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::boolean, Shape{3, 2});
        auto axes = op::Constant::create(element::i64, Shape{2}, {0, 0});
        auto reduce = std::make_shared<opset9::ReduceLogicalAnd>(data, axes, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeReduceLogicalOr) {
    {
        auto data = std::make_shared<op::Parameter>(element::boolean, Shape{1});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceLogicalOr>(data, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function = std::make_shared<Function>(
            OutputVector{std::make_shared<opset9::ReduceLogicalOr>(reduce1, reduce2_axis, true)},
            ParameterVector{data});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::boolean, Shape{1});
        auto axes = op::Constant::create(element::i64, Shape{2}, {0, 0});
        auto reduce = std::make_shared<opset9::ReduceLogicalOr>(data, axes, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeReduceMax) {
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{3, 2});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceMax>(data, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceMax>(reduce1, reduce2_axis, true)},
                                       ParameterVector{data});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{3, 2});
        auto axes = op::Constant::create(element::i64, Shape{2}, {0, 0});
        auto reduce = std::make_shared<opset9::ReduceMax>(data, axes, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeReduceMean) {
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{3, 2});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceMean>(data, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceMean>(reduce1, reduce2_axis, true)},
                                       ParameterVector{data});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{3, 2});
        auto axes = op::Constant::create(element::i64, Shape{2}, {0, 0});
        auto reduce = std::make_shared<opset9::ReduceMean>(data, axes, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeReduceMin) {
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{3, 2});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceMin>(data, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceMin>(reduce1, reduce2_axis, true)},
                                       ParameterVector{data});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{3, 2});
        auto axes = op::Constant::create(element::i64, Shape{2}, {0, 0});
        auto reduce = std::make_shared<opset9::ReduceMin>(data, axes, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeReduceProd) {
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{3, 2});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceProd>(data, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceProd>(reduce1, reduce2_axis, true)},
                                       ParameterVector{data});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{3, 2});
        auto axes = op::Constant::create(element::i64, Shape{2}, {0, 0});
        auto reduce = std::make_shared<opset9::ReduceProd>(data, axes, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeReduceSum) {
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{3, 2});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceSum>(data, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceSum>(reduce1, reduce2_axis, true)},
                                       ParameterVector{data});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{3, 2});
        auto axes = op::Constant::create(element::i64, Shape{2}, {0, 0});
        auto reduce = std::make_shared<opset9::ReduceSum>(data, axes, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeNoReduceDiffKeepDims) {
    {
        auto A = std::make_shared<op::Parameter>(element::i64, Shape{3, 2});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceL1>(reduce1, reduce2_axis, false)},
                                       ParameterVector{A});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{3, 2});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(data, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function_ref =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceL1>(reduce1, reduce2_axis, false)},
                                       ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeNotPassInvalidAxes) {
    {
        auto data = std::make_shared<op::Parameter>(element::f32, Shape{3, 2});
        auto reduce1_axis = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(data, reduce1_axis, false);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceL2>(reduce1, reduce2_axis, false)},
                                       ParameterVector{data});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::f32, Shape{3, 2});
        auto reduce1_axis = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(data, reduce1_axis, false);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function_ref =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceL2>(reduce1, reduce2_axis, false)},
                                       ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeDynamicShapes) {
    {
        auto data =
            std::make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
        auto reduce1_axis = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(data, reduce1_axis, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceL2>(reduce1, reduce2_axis, true)},
                                       ParameterVector{data});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data =
            std::make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
        auto axes = op::Constant::create(element::i64, Shape{2}, {0, 0});
        auto reduce = std::make_shared<opset9::ReduceL2>(data, axes, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMerge3ReducesL1) {
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{3, 2, 4});
        auto reduce1_axis = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(data, reduce1_axis, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {2});
        auto reduce2 = std::make_shared<opset9::ReduceL1>(reduce1, reduce2_axis, true);
        auto reduce3_axis = op::Constant::create(element::i64, Shape{1}, {1});
        auto reduce3 = std::make_shared<opset9::ReduceL1>(reduce2, reduce3_axis, true);
        function = std::make_shared<Function>(OutputVector{reduce3}, ParameterVector{data});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{3, 2, 4});
        auto axes = op::Constant::create(element::i64, Shape{3}, {0, 2, 1});
        auto reduce = std::make_shared<opset9::ReduceL1>(data, axes, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeConcatAxes) {
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{3, 2, 4});
        auto axis1 = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(data, axis1, true);
        auto axis2 = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto reduce2 = std::make_shared<opset9::ReduceL1>(reduce1, axis2, true);
        function = std::make_shared<Function>(OutputVector{reduce2}, ParameterVector{data, axis1, axis2});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{3, 2, 4});
        auto axis1 = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto axis2 = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto axes = std::make_shared<opset9::Concat>(OutputVector{axis1, axis2}, 0);
        auto reduce = std::make_shared<opset9::ReduceL1>(data, axes, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data, axis1, axis2});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}