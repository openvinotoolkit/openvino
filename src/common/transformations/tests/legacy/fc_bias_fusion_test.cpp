// Copyright (C) 2018-2022 Intel Corporation
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
#include <legacy/ngraph_ops/fully_connected.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/fc_bias_fusion.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, FullyConnectedBiasFusionTest3D) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 3072});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 3072}, {1});
        auto empty_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {0});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, empty_bias, ngraph::Shape{1, 128, 786});

        auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});

        manager.register_pass<ngraph::pass::FullyConnectedBiasFusion>();
        manager.register_pass<ngraph::pass::InjectionPass>([](std::shared_ptr<ngraph::Function> f) {
            check_rt_info(f);
        });
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 3072});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 3072}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, bias, ngraph::Shape{1, 128, 786});

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, FullyConnectedBiasFusionTest2D) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
        auto empty_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {0});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, empty_bias, ngraph::Shape{1, 786});

        auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 786}, {1});
        auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});
        manager.register_pass<ngraph::pass::FullyConnectedBiasFusion>();
        manager.register_pass<ngraph::pass::InjectionPass>([](std::shared_ptr<ngraph::Function> f) {
            check_rt_info(f);
        });
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
        auto empty_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {0});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, empty_bias, ngraph::Shape{1, 786});

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }
}


TEST_F(TransformationTestsF, FullyConnectedBiasFusionTestBias1x1) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128});

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
        auto empty_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {0});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, empty_bias, ngraph::Shape{1, 786});

        auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 1}, {1});
        auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});

        manager.register_pass<ngraph::pass::FullyConnectedBiasFusion>();
        manager.register_pass<ngraph::pass::InjectionPass>([](std::shared_ptr<ngraph::Function> function) {
            check_rt_info(function);
        });
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, bias, ngraph::Shape{1, 786});

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }
}

TEST(TransformationTests, FullyConnectedBiasFusionDynamic) {
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
    auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
    auto empty_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {0});
    auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, empty_bias, ngraph::Shape{1, 786});

    auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 786}, {1});
    auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

    auto f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::FullyConnectedBiasFusion>();
    ASSERT_NO_THROW(manager.run_passes(f));
}