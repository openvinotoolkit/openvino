// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <transformations/cpu_opset/arm/pass/convert_reduce_multi_axis.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

template <class T>
static std::shared_ptr<ov::Model> createInitGraph(std::shared_ptr<ngraph::opset1::Parameter> param) {
        auto axes = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {0, 1});
        auto reduce = std::make_shared<T>(param, axes, true);
        return std::make_shared<ngraph::Function>(ngraph::NodeVector{ reduce }, ngraph::ParameterVector{ param });
}

template <class T>
static std::shared_ptr<ov::Model> createRefGraph(ov::Shape param_shape) {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, param_shape);
        std::vector<int64_t> axes = {0, 1};
        ngraph::NodeVector new_ops;
        std::shared_ptr<ngraph::Node> node = param;
        for (auto axis : axes) {
            auto reduction_axis = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {axis});
            node = std::make_shared<T>(node, reduction_axis, true);
            new_ops.push_back(node);
        }
        auto reshape_shape = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{param_shape.size()}, {1, 1, 2, 9});
        auto reshape = std::make_shared<ngraph::opset1::Reshape>(node, reshape_shape, true);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape }, ngraph::ParameterVector{ param });
}

static ngraph::Shape static_param_shape = ngraph::Shape{2, 19, 2, 9};
static ngraph::PartialShape dynamic_param_shape = ngraph::PartialShape{2, -1, 2, 9};

TEST(TransformationTests, CheckConvertReduceMinTransformationIsApplied) {
    std::shared_ptr<ov::Model> function(nullptr), function_ref(nullptr);
    {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, static_param_shape);
        function = createInitGraph<ngraph::opset1::ReduceMin>(param);
        ov::pass::Manager manager;
        manager.register_pass<ConvertReduceMin>();
        manager.run_passes(function);
    }
    {
        function_ref = createRefGraph<ngraph::opset1::ReduceMin>(static_param_shape);
    }
    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CheckConvertReduceMaxTransformationIsApplied) {
    std::shared_ptr<ov::Model> function(nullptr), function_ref(nullptr);
    {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, static_param_shape);
        function = createInitGraph<ngraph::opset1::ReduceMax>(param);
        ov::pass::Manager manager;
        manager.register_pass<ConvertReduceMax>();
        manager.run_passes(function);
    }
    {
        function_ref = createRefGraph<ngraph::opset1::ReduceMax>(static_param_shape);
    }
    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CheckConvertReduceSumTransformationIsApplied) {
    std::shared_ptr<ov::Model> function(nullptr), function_ref(nullptr);
    {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, static_param_shape);
        function = createInitGraph<ngraph::opset1::ReduceSum>(param);
        ov::pass::Manager manager;
        manager.register_pass<ConvertReduceSum>();
        manager.run_passes(function);
    }
    {
        function_ref = createRefGraph<ngraph::opset1::ReduceSum>(static_param_shape);
    }
    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CheckConvertReduceProdTransformationIsApplied) {
    std::shared_ptr<ov::Model> function(nullptr), function_ref(nullptr);
    {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, static_param_shape);
        function = createInitGraph<ngraph::opset1::ReduceProd>(param);
        ov::pass::Manager manager;
        manager.register_pass<ConvertReduceProd>();
        manager.run_passes(function);
    }
    {
        function_ref = createRefGraph<ngraph::opset1::ReduceProd>(static_param_shape);
    }
    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CheckConvertReduceMinTransformationIsNotAppliedForDynaimcShapes) {
    std::shared_ptr<ov::Model> function(nullptr), function_ref(nullptr);
    {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, dynamic_param_shape);
        function = createInitGraph<ngraph::opset1::ReduceMin>(param);
        ov::pass::Manager manager;
        manager.register_pass<ConvertReduceMin>();
        manager.run_passes(function);
    }
    {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, dynamic_param_shape);
        function_ref = createInitGraph<ngraph::opset1::ReduceMin>(param);
    }
    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CheckConvertReduceMaxTransformationIsNotAppliedForDynaimcShapes) {
    std::shared_ptr<ov::Model> function(nullptr), function_ref(nullptr);
    {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, dynamic_param_shape);
        function = createInitGraph<ngraph::opset1::ReduceMax>(param);
        ov::pass::Manager manager;
        manager.register_pass<ConvertReduceMax>();
        manager.run_passes(function);
    }
    {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, dynamic_param_shape);
        function_ref = createInitGraph<ngraph::opset1::ReduceMax>(param);
    }
    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CheckConvertReduceSumTransformationIsNotAppliedForDynaimcShapes) {
    std::shared_ptr<ov::Model> function(nullptr), function_ref(nullptr);
    {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, dynamic_param_shape);
        function = createInitGraph<ngraph::opset1::ReduceSum>(param);
        ov::pass::Manager manager;
        manager.register_pass<ConvertReduceSum>();
        manager.run_passes(function);
    }
    {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, dynamic_param_shape);
        function_ref = createInitGraph<ngraph::opset1::ReduceSum>(param);
    }
    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CheckConvertReduceProdTransformationIsNotAppliedForDynaimcShapes) {
    std::shared_ptr<ov::Model> function(nullptr), function_ref(nullptr);
    {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, dynamic_param_shape);
        function = createInitGraph<ngraph::opset1::ReduceProd>(param);
        ov::pass::Manager manager;
        manager.register_pass<ConvertReduceProd>();
        manager.run_passes(function);
    }
    {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, dynamic_param_shape);
        function_ref = createInitGraph<ngraph::opset1::ReduceProd>(param);
    }
    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}
