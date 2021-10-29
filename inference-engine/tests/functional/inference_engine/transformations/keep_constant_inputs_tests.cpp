// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <cpp/ie_cnn_network.h>
#include <legacy/cnn_network_impl.hpp>  // deprecated API
#include <legacy/details/ie_cnn_network_iterator.hpp>  // deprecated API

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/convolution_ie.hpp>
#include <transformations/init_node_info.hpp>

#include <ie_precision.hpp>
#include <functional_test_utils/precision_utils.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include <legacy/convert_function_to_cnn_network.hpp>
#include <legacy/ngraph_ops/fully_connected.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <transformations/opset_conversions/convert_opset2_to_opset1.hpp>
#include <transformations/opset_conversions/convert_opset3_to_opset2.hpp>
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

using namespace testing;
using namespace InferenceEngine;

int numberOfInputsForLayerInCNNNetwork(const InferenceEngine::CNNNetwork & network, std::string layerType) {
    int numberOfInputs = 0;

    IE_SUPPRESS_DEPRECATED_START
    for (auto it = details::CNNNetworkIterator(network); it != details::CNNNetworkIterator(); it++) {
        InferenceEngine::CNNLayerPtr layer = *it;
        if (layer->type == layerType) {
            numberOfInputs = layer->insData.size();
            break;
        }
    }
    IE_SUPPRESS_DEPRECATED_END
    return numberOfInputs;
}

void transformNetwork(InferenceEngine::CNNNetwork & clonedNetwork, bool keep_constant_inputs) {
    if (clonedNetwork.getFunction()) {
        auto nGraphFunc = clonedNetwork.getFunction();
        ngraph::pass::CommonOptimizations().run_on_function(nGraphFunc);
        ngraph::pass::ConvertOpSet3ToOpSet2().run_on_function(nGraphFunc);
        ngraph::pass::ConvertOpSet2ToOpSet1().run_on_function(nGraphFunc);
        ngraph::pass::ConvertOpSet1ToLegacy().run_on_function(nGraphFunc);
        IE_SUPPRESS_DEPRECATED_START
        clonedNetwork = InferenceEngine::CNNNetwork(
            InferenceEngine::details::convertFunctionToICNNNetwork(nGraphFunc, clonedNetwork, keep_constant_inputs));
        IE_SUPPRESS_DEPRECATED_END
    }
}

TEST(KeepConstantInputsTests, ConvertConvolutionPoolReluNetworkWithTrue) {
    std::shared_ptr <ngraph::Function> f_ptr;
    f_ptr = ngraph::builder::subgraph::makeConvPoolRelu();
    InferenceEngine::CNNNetwork network(f_ptr), originalNetwork = network;
    transformNetwork(originalNetwork, true);
    ASSERT_EQ(numberOfInputsForLayerInCNNNetwork(originalNetwork, "Convolution"), 2);
}

TEST(KeepConstantInputsTests, ConvertConvolutionPoolReluNetworkWithFalse) {
    std::shared_ptr <ngraph::Function> f_ptr;
    f_ptr = ngraph::builder::subgraph::makeConvPoolRelu();
    InferenceEngine::CNNNetwork network(f_ptr), originalNetwork = network;
    transformNetwork(originalNetwork, false);
    ASSERT_EQ(numberOfInputsForLayerInCNNNetwork(originalNetwork, "Convolution"), 1);
}

TEST(KeepConstantInputsTests, ConvertConvolutionBiasNetworkWithTrue) {
    std::shared_ptr <ngraph::Function> f_ptr;
    f_ptr = ngraph::builder::subgraph::makeConvBias();
    InferenceEngine::CNNNetwork network(f_ptr), originalNetwork = network;
    transformNetwork(originalNetwork, true);
    ASSERT_EQ(numberOfInputsForLayerInCNNNetwork(originalNetwork, "Convolution"), 3);
}

TEST(KeepConstantInputsTests, ConvertConvolutionBiasNetworkWithFalse) {
    std::shared_ptr <ngraph::Function> f_ptr;
    f_ptr = ngraph::builder::subgraph::makeConvBias();
    InferenceEngine::CNNNetwork network(f_ptr), originalNetwork = network;
    transformNetwork(originalNetwork, false);
    ASSERT_EQ(numberOfInputsForLayerInCNNNetwork(originalNetwork, "Convolution"), 1);
}

TEST(KeepConstantInputsTests, ConvertFullyConnectedNetworkWithTrue) {
    std::shared_ptr <ngraph::Function> f_ptr;
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128});
    auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
    auto empty_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {0});
    auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, empty_bias, ngraph::Shape{1, 786});
    f_ptr = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    InferenceEngine::CNNNetwork network(f_ptr), originalNetwork = network;
    transformNetwork(originalNetwork, true);
    ASSERT_EQ(numberOfInputsForLayerInCNNNetwork(originalNetwork, "FullyConnected"), 3);
}

TEST(KeepConstantInputsTests, ConvertFullyConnectedNetworkWithFalse) {
    std::shared_ptr <ngraph::Function> f_ptr;
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128});
    auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
    auto empty_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {0});
    auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, empty_bias, ngraph::Shape{1, 786});
    f_ptr = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    InferenceEngine::CNNNetwork network(f_ptr), originalNetwork = network;
    transformNetwork(originalNetwork, false);
    ASSERT_EQ(numberOfInputsForLayerInCNNNetwork(originalNetwork, "FullyConnected"), 1);
}
