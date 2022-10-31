// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <cpp/ie_cnn_network.h>

#include "common_test_utils/ngraph_test_utils.hpp"


TEST(SmartReshapeTests, MimickingSBS) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3, 4});
        auto reshape = std::make_shared<ngraph::opset5::Reshape>(input, ngraph::opset5::Constant::create(ngraph::element::i64, {2}, {6, -1}), true);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape}, ngraph::ParameterVector{input});
    }

    InferenceEngine::CNNNetwork network(f);

    auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_NO_THROW(network.setBatchSize(2));
    check_unique_names(f, unh);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({12, 4}));
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({2, 2, 3, 4}));
}

TEST(SmartReshapeTests, MimickingSBS_1) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3, 4});
        auto reshape = std::make_shared<ngraph::opset5::Reshape>(input, ngraph::opset5::Constant::create(ngraph::element::i64, {2}, {1, -1}), true);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape}, ngraph::ParameterVector{input});
    }

    InferenceEngine::CNNNetwork network(f);

    auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_NO_THROW(network.setBatchSize(2));
    check_unique_names(f, unh);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({2, 24}));
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({2, 2, 3, 4}));
}

TEST(SmartReshapeTests, MimickingSBS_2) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{2, 2, 3, 4});
        auto reshape = std::make_shared<ngraph::opset5::Reshape>(input, ngraph::opset5::Constant::create(ngraph::element::i64, {2}, {12, -1}), true);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape}, ngraph::ParameterVector{input});
    }

    InferenceEngine::CNNNetwork network(f);

    auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_NO_THROW(network.setBatchSize(1));
    check_unique_names(f, unh);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({6, 4}));
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({1, 2, 3, 4}));
}