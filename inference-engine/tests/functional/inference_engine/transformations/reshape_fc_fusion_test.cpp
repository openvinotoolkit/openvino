// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/reshape.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <legacy/ngraph_ops/fully_connected.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/reshape_fc_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ReshapeFCFusiuonTest1) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 3, 64, 64}, {1});
        auto reshape_shape = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 3 * 64 * 64});
        auto fc_weights = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{6, 3 * 64 * 64}, {1});
        auto fc_biases = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{6}, {1});

        auto reshape = std::make_shared<ngraph::op::v1::Reshape>(input, reshape_shape, true);
        auto fc = std::make_shared<ngraph::op::FullyConnected>(reshape, fc_weights, fc_biases, ngraph::Shape{1, 6});

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ReshapeFullyConnectedFusion().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    ASSERT_EQ(f->get_ops().size(), 5);
}

TEST(TransformationTests, ReshapeFCFusiuonTest2) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 3, 64, 64}, {1});
        auto reshape_shape = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 3, 64, 64});
        auto fc_weights = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{6, 3 * 64 * 64}, {1});
        auto fc_biases = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{6}, {1});

        auto reshape = std::make_shared<ngraph::op::v1::Reshape>(input, reshape_shape, true);
        auto fc = std::make_shared<ngraph::op::FullyConnected>(reshape, fc_weights, fc_biases, ngraph::Shape{1, 6});

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ReshapeFullyConnectedFusion().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    ASSERT_EQ(f->get_ops().size(), 5);
}

TEST(TransformationTests, ReshapeFCFusiuonTest3) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{2, 3, 64, 64}, {1});
        auto reshape_shape = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 2 * 3 * 64 * 64});
        auto fc_weights = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{6, 2 * 3 * 64 * 64}, {1});
        auto fc_biases = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{6}, {1});

        auto reshape = std::make_shared<ngraph::op::v1::Reshape>(input, reshape_shape, true);
        auto fc = std::make_shared<ngraph::op::FullyConnected>(reshape, fc_weights, fc_biases, ngraph::Shape{2, 6});

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ReshapeFullyConnectedFusion().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    ASSERT_EQ(f->get_ops().size(), 7);
}
