// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cpp/ie_cnn_network.h>
#include <cnn_network_impl.hpp>  // deprecated API

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/convolution_ie.hpp>
#include <transformations/init_node_info.hpp>

#include <ie_precision.hpp>
#include <functional_test_utils/precision_utils.hpp>
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "low_precision_transformations/network_helper.hpp"


using namespace testing;
using namespace InferenceEngine;

bool isCNNNetworkQuantized(const InferenceEngine::CNNNetwork& network) {
    CNNLayerPtr layerPtr = nullptr;
    layerPtr = details::CNNNetworkHelper::getLayer(network, "FakeQuantize");
    if (layerPtr != nullptr)
        return true;
    else
        return false;
}

TEST(KeepConstantInputsTests, ConvertConvolutionNetwork) {
    std::shared_ptr <ngraph::Function> f;
    {
        auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 64, 64});
        auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1});
        auto convolution = std::make_shared<ngraph::op::ConvolutionIE>(param1, param2,
                                                                       ngraph::Strides{1, 1},
                                                                       ngraph::Strides{1, 1},
                                                                       ngraph::CoordinateDiff{0, 0},
                                                                       ngraph::CoordinateDiff{0, 0});
        convolution->set_friendly_name("convolution");
        auto result = std::make_shared<ngraph::op::Result>(convolution);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                               ngraph::ParameterVector{param1, param2});

        ngraph::pass::InitNodeInfo().run_on_function(f);
    }

    InferenceEngine::CNNNetwork nGraphImpl(f);

    ASSERT_FALSE(isCNNNetworkQuantized(nGraphImpl));

//    try {
//        auto net = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(
//                                                        static_cast<const InferenceEngine::ICNNNetwork &>(nGraphImpl));
//    } catch (InferenceEngine::details::InferenceEngineException &err) {
//        FAIL();
//    }
}


TEST(KeepConstantInputsTests, ConvertConvolutionPoolReluNetwork) {
    std::shared_ptr <ngraph::Function> f_ptr;

    f_ptr = ngraph::builder::subgraph::makeConvPoolRelu();

    ngraph::pass::InitNodeInfo().run_on_function(f_ptr);

    InferenceEngine::CNNNetwork nGraphImpl(f_ptr);

    ASSERT_FALSE(isCNNNetworkQuantized(nGraphImpl));

//    try {
//    auto net = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(
//                                                        static_cast<const InferenceEngine::ICNNNetwork &>(nGraphImpl));
//    } catch (InferenceEngine::details::InferenceEngineException &err) {
//        FAIL();
//    }
}

TEST(KeepConstantInputsTests, ConvertConvBiasNetwork) {
std::shared_ptr <ngraph::Function> f_ptr;

f_ptr = ngraph::builder::subgraph::makeConvBias();

ngraph::pass::InitNodeInfo().run_on_function(f_ptr);

InferenceEngine::CNNNetwork nGraphImpl(f_ptr);

ASSERT_FALSE(isCNNNetworkQuantized(nGraphImpl));

//    try {
//    auto net = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(
//                                                        static_cast<const InferenceEngine::ICNNNetwork &>(nGraphImpl));
//    } catch (InferenceEngine::details::InferenceEngineException &err) {
//        FAIL();
//    }
}
