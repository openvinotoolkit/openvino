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
#include <transformations/cpu_opset/arm/pass/convert_group_conv1d.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

template <class T>
static std::shared_ptr<ov::Model> createInitGraph(ngraph::Shape param_shape, ngraph::Shape weights_shape) {
        auto type = ngraph::element::f32;
        auto param = std::make_shared<ngraph::opset1::Parameter>(type, param_shape);
        auto weights = ngraph::opset1::Constant::create(type, weights_shape, { 1 });
        bool is1Dinput = param_shape.size() == 3;
        auto conv = std::make_shared<T>(param,
                                        weights,
                                        is1Dinput ? ngraph::Strides{1} :        ngraph::Strides{1, 1},
                                        is1Dinput ? ngraph::CoordinateDiff{0} : ngraph::CoordinateDiff{0, 0},
                                        is1Dinput ? ngraph::CoordinateDiff{0} : ngraph::CoordinateDiff{0, 0},
                                        is1Dinput ? ngraph::Strides{1} :        ngraph::Strides{1, 1});

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{ conv }, ngraph::ParameterVector{ param });
}

template <class T>
static std::shared_ptr<ov::Model> createTransformedGraph(ngraph::Shape param_shape, ngraph::Shape weights_shape) {
        auto getUnsqueeze = [&](const ngraph::Output<ngraph::Node>& node) {
            auto rank = node.get_partial_shape().rank().get_length();
            return std::make_shared<ov::opset8::Unsqueeze>(node,
                                                           ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {rank}));
        };
        auto type = ngraph::element::f32;
        auto param = std::make_shared<ngraph::opset1::Parameter>(type, param_shape);
        auto weights = ngraph::opset1::Constant::create(type, weights_shape, { 1 });
        auto input2d = getUnsqueeze(param);
        auto weights2d = getUnsqueeze(weights);
        auto conv2d = std::make_shared<T>(input2d,
                                          weights2d,
                                          ngraph::Strides{1, 1},
                                          ngraph::CoordinateDiff{0, 0},
                                          ngraph::CoordinateDiff{0, 0},
                                          ngraph::Strides{1, 1});

        auto reshape = std::make_shared<ngraph::opset1::Squeeze>(conv2d,
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {3}));
        return std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape }, ngraph::ParameterVector{ param });
}

TEST(TransformationTests, CheckConvertConv1DIsAppliedFor1DShapes) {
    std::shared_ptr<ov::Model> function(nullptr), function_ref(nullptr);
    {
        function = createInitGraph<ngraph::opset1::Convolution>(ngraph::Shape{2, 64, 7}, ngraph::Shape{ 30, 64, 1 });
        ov::pass::Manager manager;
        manager.register_pass<ConvertConv1D>();
        manager.run_passes(function);
    }
    {
        function_ref = createTransformedGraph<ngraph::opset1::Convolution>(ngraph::Shape{2, 64, 7}, ngraph::Shape{30, 64, 1});
    }
    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CheckConvertConv1DIsNotAppliedFor2DShapes) {
    std::shared_ptr<ov::Model> function(nullptr), function_ref(nullptr);
    {
        function = createInitGraph<ngraph::opset1::Convolution>(ngraph::Shape{2, 64, 7, 1}, ngraph::Shape{30, 64, 1, 1});
        ov::pass::Manager manager;
        manager.register_pass<ConvertConv1D>();
        manager.run_passes(function);
    }
    {
        function_ref = createInitGraph<ngraph::opset1::Convolution>(ngraph::Shape{2, 64, 7, 1}, ngraph::Shape{30, 64, 1, 1});
    }
    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CheckConvertGroupConv1DIsAppliedFor1dShapes) {
    std::shared_ptr<ov::Model> function(nullptr), function_ref(nullptr);
    {
        function = createInitGraph<ngraph::opset1::GroupConvolution>(ngraph::Shape{1, 12, 64}, ngraph::Shape{4, 1, 3, 5});
        ov::pass::Manager manager;
        manager.register_pass<ConvertGroupConv1D>();
        manager.run_passes(function);
    }
    {
        function_ref = createTransformedGraph<ngraph::opset1::GroupConvolution>(ngraph::Shape{1, 12, 64}, ngraph::Shape{4, 1, 3, 5});
    }
    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CheckConvertGroupConv1DIsNotAppliedFor2DShapes) {
    std::shared_ptr<ov::Model> function(nullptr), function_ref(nullptr);
    {
        function = createInitGraph<ngraph::opset1::GroupConvolution>(ngraph::Shape{1, 12, 64, 1}, ngraph::Shape{4, 1, 3, 5, 1});
        ov::pass::Manager manager;
        manager.register_pass<ConvertGroupConv1D>();
        manager.run_passes(function);
    }
    {
        function_ref = createInitGraph<ngraph::opset1::GroupConvolution>(ngraph::Shape{1, 12, 64, 1}, ngraph::Shape{4, 1, 3, 5, 1});
    }
    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}
