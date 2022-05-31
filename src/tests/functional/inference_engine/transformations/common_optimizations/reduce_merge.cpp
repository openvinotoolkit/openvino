// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/reduce_merge.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace ngraph;

TEST_F(TransformationTestsF, ReduceMergeReduceL1) {
    {
        Shape shape{1};
        auto type = element::f32;
        auto A = std::make_shared<op::Parameter>(type, shape);
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceL1>(reduce1, reduce2_axis, true)},
                                       ParameterVector{A});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::f32, Shape{1});
        auto axis1 = op::Constant::create(element::i64, Shape{1}, {0});
        auto axis2 = op::Constant::create(element::i64, Shape{1}, {0});
        auto concat = std::make_shared<opset9::Concat>(OutputVector{axis1, axis2}, 0);
        auto reduce = std::make_shared<opset9::ReduceL1>(data, concat, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ReduceMergeReduceL2) {
    {
        Shape shape{1};
        auto type = element::f32;
        auto A = std::make_shared<op::Parameter>(type, shape);
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceL2>(reduce1, reduce2_axis, true)},
                                       ParameterVector{A});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::f32, Shape{1});
        auto axis1 = op::Constant::create(element::i64, Shape{1}, {0});
        auto axis2 = op::Constant::create(element::i64, Shape{1}, {0});
        auto concat = std::make_shared<opset9::Concat>(OutputVector{axis1, axis2}, 0);
        auto reduce = std::make_shared<opset9::ReduceL2>(data, concat, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ReduceMergeReduceLogicalAnd) {
    {
        auto A = std::make_shared<op::Parameter>(element::boolean, Shape{1});
        auto reduce1_axis = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceLogicalAnd>(A, reduce1_axis, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function = std::make_shared<Function>(
            OutputVector{std::make_shared<opset9::ReduceLogicalAnd>(reduce1, reduce2_axis, true)},
            ParameterVector{A});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::boolean, Shape{1});
        auto axis1 = op::Constant::create(element::i64, Shape{1}, {0});
        auto axis2 = op::Constant::create(element::i64, Shape{1}, {0});
        auto concat = std::make_shared<opset9::Concat>(OutputVector{axis1, axis2}, 0);
        auto reduce = std::make_shared<opset9::ReduceLogicalAnd>(data, concat, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ReduceMergeReduceLogicalOr) {
    {
        auto A = std::make_shared<op::Parameter>(element::boolean, Shape{1});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceLogicalOr>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function = std::make_shared<Function>(
            OutputVector{std::make_shared<opset9::ReduceLogicalOr>(reduce1, reduce2_axis, true)},
            ParameterVector{A});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::boolean, Shape{1});
        auto axis1 = op::Constant::create(element::i64, Shape{1}, {0});
        auto axis2 = op::Constant::create(element::i64, Shape{1}, {0});
        auto concat = std::make_shared<opset9::Concat>(OutputVector{axis1, axis2}, 0);
        auto reduce = std::make_shared<opset9::ReduceLogicalOr>(data, concat, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ReduceMergeReduceMax) {
    {
        auto A = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceMax>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceMax>(reduce1, reduce2_axis, true)},
                                       ParameterVector{A});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto axis1 = op::Constant::create(element::i64, Shape{1}, {0});
        auto axis2 = op::Constant::create(element::i64, Shape{1}, {0});
        auto concat = std::make_shared<opset9::Concat>(OutputVector{axis1, axis2}, 0);
        auto reduce = std::make_shared<opset9::ReduceMax>(data, concat, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ReduceMergeReduceMean) {
    {
        auto A = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceMean>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceMean>(reduce1, reduce2_axis, true)},
                                       ParameterVector{A});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto axis1 = op::Constant::create(element::i64, Shape{1}, {0});
        auto axis2 = op::Constant::create(element::i64, Shape{1}, {0});
        auto concat = std::make_shared<opset9::Concat>(OutputVector{axis1, axis2}, 0);
        auto reduce = std::make_shared<opset9::ReduceMean>(data, concat, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ReduceMergeReduceMin) {
    {
        auto A = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceMin>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceMin>(reduce1, reduce2_axis, true)},
                                       ParameterVector{A});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto axis1 = op::Constant::create(element::i64, Shape{1}, {0});
        auto axis2 = op::Constant::create(element::i64, Shape{1}, {0});
        auto concat = std::make_shared<opset9::Concat>(OutputVector{axis1, axis2}, 0);
        auto reduce = std::make_shared<opset9::ReduceMin>(data, concat, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ReduceProd) {
    {
        auto A = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceProd>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceProd>(reduce1, reduce2_axis, true)},
                                       ParameterVector{A});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto axis1 = op::Constant::create(element::i64, Shape{1}, {0});
        auto axis2 = op::Constant::create(element::i64, Shape{1}, {0});
        auto concat = std::make_shared<opset9::Concat>(OutputVector{axis1, axis2}, 0);
        auto reduce = std::make_shared<opset9::ReduceProd>(data, concat, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ReduceMergeReduceSum) {
    {
        auto A = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceSum>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceSum>(reduce1, reduce2_axis, true)},
                                       ParameterVector{A});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto axis1 = op::Constant::create(element::i64, Shape{1}, {0});
        auto axis2 = op::Constant::create(element::i64, Shape{1}, {0});
        auto concat = std::make_shared<opset9::Concat>(OutputVector{axis1, axis2}, 0);
        auto reduce = std::make_shared<opset9::ReduceSum>(data, concat, true);
        function_ref = std::make_shared<Function>(OutputVector{reduce}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ReduceMergeNoReduceDiffKeepDims) {
    {
        auto A = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceL1>(reduce1, reduce2_axis, false)},
                                       ParameterVector{A});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(data, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function_ref =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceL1>(reduce1, reduce2_axis, false)},
                                       ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ReduceMergeNoReduceMergeInvalidAxes) {
    {
        Shape shape{2, 1};
        auto type = element::f32;
        auto A = std::make_shared<op::Parameter>(type, shape);
        auto reduce1_axis = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(A, reduce1_axis, false);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceL2>(reduce1, reduce2_axis, false)},
                                       ParameterVector{A});
        manager.register_pass<pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::Parameter>(element::f32, Shape{2, 1});
        auto reduce1_axis = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(data, reduce1_axis, false);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        function_ref =
            std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceL2>(reduce1, reduce2_axis, false)},
                                       ParameterVector{data});
    }
}
