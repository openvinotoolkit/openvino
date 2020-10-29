// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "tests_common.hpp"

#include <convert_function_to_cnn_network.hpp>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/convolution_ie.hpp>

using namespace testing;
using namespace InferenceEngine;

using ConvertFunctionToCNNNetworkTests = TestsCommon;

TEST_F(ConvertFunctionToCNNNetworkTests, ConvertPReLUNetwork) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 2});
        auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 2});
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(param1, param2);
        prelu->set_friendly_name("prelu");
        auto result = std::make_shared<ngraph::op::Result>(prelu);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                               ngraph::ParameterVector{param1, param2});
    }

    InferenceEngine::details::CNNNetworkNGraphImpl nGraphImpl(f);
    try {
        auto net = InferenceEngine::details::convertFunctionToICNNNetwork(f, nGraphImpl);
        FAIL();
    } catch (InferenceEngine::details::InferenceEngineException &err) {
        const std::string ref_msg = "Error of validate layer: prelu with type: PReLU. Number of inputs (2) is not equal to expected ones: 1";
        const std::string resp_msg = err.what();
        ASSERT_TRUE(resp_msg.find(ref_msg) != std::string::npos);
    }
}

TEST_F(ConvertFunctionToCNNNetworkTests, ConvertConvolutionNetwork) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 64, 64});
        auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1});
        auto convolution = std::make_shared<ngraph::op::ConvolutionIE>(param1, param2,
                                                                  ngraph::Strides{1, 1},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::Strides{1, 1},
                                                                  ngraph::Shape{1, 3, 64, 64});
        convolution->set_friendly_name("convolution");
        auto result = std::make_shared<ngraph::op::Result>(convolution);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                               ngraph::ParameterVector{param1, param2});
    }

    InferenceEngine::details::CNNNetworkNGraphImpl nGraphImpl(f);
    try {
        auto net = InferenceEngine::details::convertFunctionToICNNNetwork(f, nGraphImpl);
    } catch (InferenceEngine::details::InferenceEngineException &err) {
        FAIL();
    }
}