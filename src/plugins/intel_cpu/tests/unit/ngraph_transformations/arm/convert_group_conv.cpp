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
#include <transformations/cpu_opset/arm/pass/convert_group_conv.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

template <class T>
static std::shared_ptr<ov::Model> createInitGraph(std::shared_ptr<ngraph::opset1::Parameter> param, ngraph::Shape weights_shape) {
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, weights_shape, { 1 });
        auto conv = std::make_shared<T>(param,
                                        weights,
                                        ngraph::Strides{1},
                                        ngraph::CoordinateDiff{0},
                                        ngraph::CoordinateDiff{0},
                                        ngraph::Strides{1});

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{ conv }, ngraph::ParameterVector{ param });
}

TEST(TransformationTests, CheckConvertGroupConvIsApplied) {
    std::shared_ptr<ov::Model> function(nullptr), function_ref(nullptr);
    {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 6, 224});
        function = createInitGraph<ngraph::opset1::GroupConvolution>(param, ngraph::Shape{2, 1, 3, 5});
        ov::pass::Manager manager;
        manager.register_pass<ConvertGroupConvolution>();
        manager.run_passes(function);
    }
    {
        const unsigned int groups = 2;
        const unsigned int channel_axis = 1;
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 6, 224});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{groups, 1, 3, 5}, { 1 });
        auto split_weights = std::make_shared<ngraph::opset1::Split>(weights,
                                                                     ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0}),
                                                                     groups);
        auto axis  = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {channel_axis});
        auto split = std::make_shared<ngraph::opset1::Split>(param, axis, groups);
        ngraph::NodeVector concat_inputs;
        for (size_t g = 0; g < groups; g++) {
            auto out = split->output(g);
            auto filter = std::make_shared<ngraph::opset1::Squeeze>(split_weights->output(g),
                                                                    ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0}));
            auto conv = std::make_shared<ngraph::opset1::Convolution>(out,
                                                                      filter,
                                                                      ngraph::Strides{1},
                                                                      ngraph::CoordinateDiff{0},
                                                                      ngraph::CoordinateDiff{0},
                                                                      ngraph::Strides{1});
            concat_inputs.push_back(conv);
        }
        auto concat = std::make_shared<ov::opset8::Concat>(concat_inputs, 1);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ param });
    }
    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CheckConvertGroupConvIsNotAppliedForDepthwiseCase) {
    std::shared_ptr<ov::Model> function(nullptr), function_ref(nullptr);
    {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 224});
        function = createInitGraph<ngraph::opset1::GroupConvolution>(param, ngraph::Shape{2, 1, 1, 5});
        ov::pass::Manager manager;
        manager.register_pass<ConvertGroupConvolution>();
        manager.run_passes(function);
    }
    {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 224});
        function_ref = createInitGraph<ngraph::opset1::GroupConvolution>(param, ngraph::Shape{2, 1, 1, 5});
    }
    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CheckConvertGroupConvIsNotAppliedForDynamicShapes) {
    std::shared_ptr<ov::Model> function(nullptr), function_ref(nullptr);
    {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, -1, 224});
        function = createInitGraph<ngraph::opset1::GroupConvolution>(param, ngraph::Shape{2, 1, 1, 5});
        ov::pass::Manager manager;
        manager.register_pass<ConvertGroupConvolution>();
        manager.run_passes(function);
    }
    {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, -1, 224});
        function_ref = createInitGraph<ngraph::opset1::GroupConvolution>(param, ngraph::Shape{2, 1, 1, 5});
    }
    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}