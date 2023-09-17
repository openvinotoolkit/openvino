// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <legacy/ngraph_ops/fully_connected.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/fc_bias_fusion.hpp>
#include <map>
#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <queue>
#include <sstream>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"

using namespace testing;

TEST_F(TransformationTestsF, FullyConnectedBiasFusionTest3D) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 128, 3072});
        auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786, 3072}, {1});
        auto empty_bias = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786}, {0});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, empty_bias, ov::Shape{1, 128, 786});

        auto const_bias = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786}, {1});
        auto add = std::make_shared<ov::opset1::Add>(fc, const_bias);

        model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input1});

        manager.register_pass<ngraph::pass::FullyConnectedBiasFusion>();
        manager.register_pass<ov::pass::InjectionPass>([](std::shared_ptr<ov::Model> f) {
            check_rt_info(f);
        });
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 128, 3072});
        auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786, 3072}, {1});
        auto bias = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786}, {1});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, bias, ov::Shape{1, 128, 786});

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, FullyConnectedBiasFusionTest2D) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 128});
        auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786, 128}, {1});
        auto empty_bias = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786}, {0});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, empty_bias, ov::Shape{1, 786});

        auto const_bias = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 786}, {1});
        auto add = std::make_shared<ov::opset1::Add>(fc, const_bias);

        model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input1});
        manager.register_pass<ngraph::pass::FullyConnectedBiasFusion>();
        manager.register_pass<ov::pass::InjectionPass>([](std::shared_ptr<ov::Model> f) {
            check_rt_info(f);
        });
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 128});
        auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786, 128}, {1});
        auto empty_bias = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786}, {0});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, empty_bias, ov::Shape{1, 786});

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, FullyConnectedBiasFusionTestBias1x1) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 128});

        auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786, 128}, {1});
        auto empty_bias = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786}, {0});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, empty_bias, ov::Shape{1, 786});

        auto const_bias = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 1}, {1});
        auto add = std::make_shared<ov::opset1::Add>(fc, const_bias);

        model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input1});

        manager.register_pass<ngraph::pass::FullyConnectedBiasFusion>();
        manager.register_pass<ov::pass::InjectionPass>([](std::shared_ptr<ov::Model> function) {
            check_rt_info(function);
        });
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 128});
        auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786, 128}, {1});
        auto bias = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786}, {1});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, bias, ov::Shape{1, 786});

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});
    }
}

TEST(TransformationTests, FullyConnectedBiasFusionDynamic) {
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786, 128}, {1});
    auto empty_bias = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786}, {0});
    auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, empty_bias, ov::Shape{1, 786});

    auto const_bias = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 786}, {1});
    auto add = std::make_shared<ov::opset1::Add>(fc, const_bias);

    auto f = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input1});
    ov::pass::Manager manager;
    manager.register_pass<ngraph::pass::FullyConnectedBiasFusion>();
    ASSERT_NO_THROW(manager.run_passes(f));
}
