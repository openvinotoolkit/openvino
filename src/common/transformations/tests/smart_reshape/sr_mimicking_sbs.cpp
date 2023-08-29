// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cpp/ie_cnn_network.h>
#include <gtest/gtest.h>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset5.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace ov;

TEST(SmartReshapeTests, MimickingSBS) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 3, 4});
        auto reshape =
            std::make_shared<opset5::Reshape>(input, opset5::Constant::create(element::i64, {2}, {6, -1}), true);
        f = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input});
    }

    InferenceEngine::CNNNetwork network(f);

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_NO_THROW(network.setBatchSize(2));
    check_unique_names(f, unh);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({12, 4}));
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({2, 2, 3, 4}));
}

TEST(SmartReshapeTests, MimickingSBS_1) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 3, 4});
        auto reshape =
            std::make_shared<opset5::Reshape>(input, opset5::Constant::create(element::i64, {2}, {1, -1}), true);
        f = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input});
    }

    InferenceEngine::CNNNetwork network(f);

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_NO_THROW(network.setBatchSize(2));
    check_unique_names(f, unh);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({2, 24}));
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({2, 2, 3, 4}));
}

TEST(SmartReshapeTests, MimickingSBS_2) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2, 3, 4});
        auto reshape =
            std::make_shared<opset5::Reshape>(input, opset5::Constant::create(element::i64, {2}, {12, -1}), true);
        f = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input});
    }

    InferenceEngine::CNNNetwork network(f);

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_NO_THROW(network.setBatchSize(1));
    check_unique_names(f, unh);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({6, 4}));
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({1, 2, 3, 4}));
}
