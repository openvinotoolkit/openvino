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

TEST(ReduceMerge, ReduceL1) {
    std::shared_ptr<Function> f;
    {
        Shape shape{1};
        auto type = element::f32;
        auto A = std::make_shared<op::Parameter>(type, shape);
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        f = std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceL1>(reduce1, reduce2_axis, true)},
                                       ParameterVector{A});
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReduceMerge>();
    pass_manager.run_passes(f);
    ASSERT_TRUE(count_ops_of_type<opset9::ReduceL1>(f) == 1);
    ASSERT_TRUE(count_ops_of_type<opset9::Concat>(f) == 1);
}

TEST(ReduceMerge, ReduceL2) {
    std::shared_ptr<Function> f;
    {
        Shape shape{1};
        auto type = element::f32;
        auto A = std::make_shared<op::Parameter>(type, shape);
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        f = std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceL2>(reduce1, reduce2_axis, true)},
                                       ParameterVector{A});
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReduceMerge>();
    pass_manager.run_passes(f);
    ASSERT_TRUE(count_ops_of_type<opset9::ReduceL2>(f) == 1);
    ASSERT_TRUE(count_ops_of_type<opset9::Concat>(f) == 1);
}

TEST(ReduceMerge, ReduceLogicalAnd) {
    std::shared_ptr<Function> f;
    {
        auto A = std::make_shared<op::Parameter>(element::boolean, Shape{1});
        auto reduce1_axis = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceLogicalAnd>(A, reduce1_axis, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        f = std::make_shared<Function>(
            OutputVector{std::make_shared<opset9::ReduceLogicalAnd>(reduce1, reduce2_axis, true)},
            ParameterVector{A});
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReduceMerge>();
    pass_manager.run_passes(f);
    ASSERT_TRUE(count_ops_of_type<opset9::ReduceLogicalAnd>(f) == 1);
    ASSERT_TRUE(count_ops_of_type<opset9::Concat>(f) == 1);
}

TEST(ReduceMerge, ReduceLogicalOr) {
    std::shared_ptr<Function> f;
    {
        auto A = std::make_shared<op::Parameter>(element::boolean, Shape{1});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceLogicalOr>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        f = std::make_shared<Function>(
            OutputVector{std::make_shared<opset9::ReduceLogicalOr>(reduce1, reduce2_axis, true)},
            ParameterVector{A});
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReduceMerge>();
    pass_manager.run_passes(f);
    ASSERT_TRUE(count_ops_of_type<opset9::ReduceLogicalOr>(f) == 1);
    ASSERT_TRUE(count_ops_of_type<opset9::Concat>(f) == 1);
}

TEST(ReduceMerge, ReduceMax) {
    std::shared_ptr<Function> f;
    {
        auto A = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceMax>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        f = std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceMax>(reduce1, reduce2_axis, true)},
                                       ParameterVector{A});
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReduceMerge>();
    pass_manager.run_passes(f);
    ASSERT_TRUE(count_ops_of_type<opset9::ReduceMax>(f) == 1);
    ASSERT_TRUE(count_ops_of_type<opset9::Concat>(f) == 1);
}

TEST(ReduceMerge, ReduceMean) {
    std::shared_ptr<Function> f;
    {
        auto A = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceMean>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        f = std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceMean>(reduce1, reduce2_axis, true)},
                                       ParameterVector{A});
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReduceMerge>();
    pass_manager.run_passes(f);
    ASSERT_TRUE(count_ops_of_type<opset9::ReduceMean>(f) == 1);
    ASSERT_TRUE(count_ops_of_type<opset9::Concat>(f) == 1);
}

TEST(ReduceMerge, ReduceMin) {
    std::shared_ptr<Function> f;
    {
        auto A = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceMin>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        f = std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceMin>(reduce1, reduce2_axis, true)},
                                       ParameterVector{A});
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReduceMerge>();
    pass_manager.run_passes(f);
    ASSERT_TRUE(count_ops_of_type<opset9::ReduceMin>(f) == 1);
    ASSERT_TRUE(count_ops_of_type<opset9::Concat>(f) == 1);
}

TEST(ReduceMerge, ReduceProd) {
    std::shared_ptr<Function> f;
    {
        auto A = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceProd>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        f = std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceProd>(reduce1, reduce2_axis, true)},
                                       ParameterVector{A});
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReduceMerge>();
    pass_manager.run_passes(f);
    ASSERT_TRUE(count_ops_of_type<opset9::ReduceProd>(f) == 1);
    ASSERT_TRUE(count_ops_of_type<opset9::Concat>(f) == 1);
}

TEST(ReduceMerge, ReduceSum) {
    std::shared_ptr<Function> f;
    {
        auto A = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceSum>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        f = std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceSum>(reduce1, reduce2_axis, true)},
                                       ParameterVector{A});
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReduceMerge>();
    pass_manager.run_passes(f);
    ASSERT_TRUE(count_ops_of_type<opset9::ReduceSum>(f) == 1);
    ASSERT_TRUE(count_ops_of_type<opset9::Concat>(f) == 1);
}

TEST(ReduceMerge, NoReduceDifferentKeepDims) {
    std::shared_ptr<Function> f;
    {
        auto A = std::make_shared<op::Parameter>(element::i64, Shape{1});
        auto reduce1_axes = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(A, reduce1_axes, true);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        f = std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceL1>(reduce1, reduce2_axis, false)},
                                       ParameterVector{A});
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReduceMerge>();
    pass_manager.run_passes(f);
    ASSERT_TRUE(count_ops_of_type<opset9::ReduceL1>(f) == 2);
    ASSERT_TRUE(count_ops_of_type<opset9::Concat>(f) == 0);
}

TEST(ReduceMerge, NoReduceMergeInvalidAxes) {
    std::shared_ptr<Function> f;
    {
        Shape shape{2, 1};
        auto type = element::f32;
        auto A = std::make_shared<op::Parameter>(type, shape);
        auto reduce1_axis = op::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(A, reduce1_axis, false);
        auto reduce2_axis = op::Constant::create(element::i64, Shape{1}, {0});
        f = std::make_shared<Function>(OutputVector{std::make_shared<opset9::ReduceL2>(reduce1, reduce2_axis, false)},
                                       ParameterVector{A});
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReduceMerge>();
    pass_manager.run_passes(f);
    ASSERT_TRUE(count_ops_of_type<opset9::ReduceL2>(f) == 2);
    ASSERT_TRUE(count_ops_of_type<opset9::Concat>(f) == 0);
}
