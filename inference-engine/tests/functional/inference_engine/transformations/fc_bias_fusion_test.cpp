// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph_ops/fully_connected.hpp>
#include <transformations/convert_opset1_to_legacy/fc_bias_fusion.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>

#include "ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, FullyConnectedBiasFusionTest3D) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 3072});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 3072}, {1});
        auto empty_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {0});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, empty_bias, ngraph::Shape{1, 128, 786});

        auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::FullyConnectedBiasFusion().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ngraph::pass::ConstantFolding().run_on_function(f);
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 3072});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 3072}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, bias, ngraph::Shape{1, 128, 786});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, FullyConnectedBiasFusionTest2D) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
        auto empty_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {0});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, empty_bias, ngraph::Shape{1, 786});

        auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 786}, {1});
        auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::FullyConnectedBiasFusion().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ngraph::pass::ConstantFolding().run_on_function(f);
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
        auto empty_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {0});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, empty_bias, ngraph::Shape{1, 786});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
