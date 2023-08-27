// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cpp/ie_cnn_network.h>
#include <gtest/gtest.h>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset5.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace ov;

TEST(SmartReshapeTests, Reshape1d) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto reshape = std::make_shared<opset5::Reshape>(input, opset5::Constant::create(element::i64, {1}, {5}), true);
        f = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input});
    }

    InferenceEngine::CNNNetwork network(f);

    ASSERT_TRUE(
        network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible(PartialShape::dynamic()));
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({5}));

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_NO_THROW(network.reshape(
        InferenceEngine::ICNNNetwork::InputShapes{{f->get_parameters()[0]->get_friendly_name(), {1, 3, 300, 300}}}));
    check_unique_names(f, unh);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({270000}));
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({1, 3, 300, 300}));
}

TEST(SmartReshapeTests, Reshape1d_negative) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto pattern = std::make_shared<opset5::Parameter>(element::i64, Shape{1});
        auto reshape = std::make_shared<opset5::Reshape>(input, pattern, false);
        f = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input, pattern});
    }

    InferenceEngine::CNNNetwork network(f);

    ASSERT_TRUE(
        network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible(PartialShape::dynamic()));
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().is_dynamic());

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_NO_THROW(network.reshape(
        InferenceEngine::ICNNNetwork::InputShapes{{f->get_parameters()[0]->get_friendly_name(), {1, 3, 300, 300}}}));
    check_unique_names(f, unh);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({270000}));
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({1, 3, 300, 300}));
    ASSERT_FALSE(network.getFunction()->get_parameters()[1]->get_output_target_inputs(0).empty());
}
