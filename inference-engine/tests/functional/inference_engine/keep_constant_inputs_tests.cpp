// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <iostream>
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
#include "convert_function_to_cnn_network.hpp"
#include "graph_tools.hpp"

using namespace testing;
using namespace InferenceEngine;

bool isInputConstLayersInCNNNetwork(const InferenceEngine::CNNNetwork& network, const std::string& layerType) {
    int numberOfInputs = 0;
    auto layersInNetwork = network.getFunction()->get_ordered_ops();
    for (auto layer : layersInNetwork) {
        if (std::string(layer->get_type_name()) == layerType) {
            std::cout << "Found layer " << layerType << std::endl;
            numberOfInputs = layer->get_input_size();
            std::cout << "Number of inputs in " << layerType << " = " << numberOfInputs << std::endl;
            break;
        }
    }
    return  numberOfInputs > 1;
}

TEST(KeepConstantInputsTests, ConvertConvolutionPoolReluNetworkWithTrue) {
    std::shared_ptr <ngraph::Function> f_ptr;
    f_ptr = ngraph::builder::subgraph::makeConvPoolRelu();
    InferenceEngine::CNNNetwork originalNetwork(f_ptr);
    std::shared_ptr<ICNNNetwork> ptrToConvertedNetwork;
//    ptrToConvertedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(f_ptr, originalNetwork, true);
//    InferenceEngine::CNNNetwork convertedNetwork(ptrToConvertedNetwork->getFunction());

    ASSERT_TRUE(isInputConstLayersInCNNNetwork(originalNetwork, "Convolution"));
//    ASSERT_TRUE(isInputConstLayersInCNNNetwork(convertedNetwork, "Convvolution"));
}

//TEST(KeepConstantInputsTests, ConvertConvolutionPoolReluNetworkWithFalse) {
//    std::shared_ptr <ngraph::Function> f_ptr;
//    f_ptr = ngraph::builder::subgraph::makeConvPoolRelu();
//    ngraph::pass::InitNodeInfo().run_on_function(f_ptr);
//    InferenceEngine::CNNNetwork nGraphImpl(f_ptr);
//    auto originalNetwork = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(
//            static_cast<const InferenceEngine::ICNNNetwork &>(nGraphImpl));
//    std::shared_ptr<ICNNNetwork> convertedNetwork;
//    convertedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(f_ptr, originalNetwork, false);
//}
//
//
//TEST(KeepConstantInputsTests, ConvertConvBiasNetworkWithTrue) {
//    std::shared_ptr <ngraph::Function> f_ptr;
//    f_ptr = ngraph::builder::subgraph::makeConvBias();
//    ngraph::pass::InitNodeInfo().run_on_function(f_ptr);
//    InferenceEngine::CNNNetwork nGraphImpl(f_ptr);
//    auto originalNetwork = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(
//                                        static_cast<const InferenceEngine::ICNNNetwork &>(nGraphImpl));
//    std::shared_ptr<ICNNNetwork> convertedNetwork;
//    convertedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(f_ptr, originalNetwork, true);
//}
//
//TEST(KeepConstantInputsTests, ConvertConvBiasNetworkWithFalse) {
//std::shared_ptr <ngraph::Function> f_ptr;
//f_ptr = ngraph::builder::subgraph::makeConvBias();
//ngraph::pass::InitNodeInfo().run_on_function(f_ptr);
//InferenceEngine::CNNNetwork nGraphImpl(f_ptr);
//auto originalNetwork = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(
//        static_cast<const InferenceEngine::ICNNNetwork &>(nGraphImpl));
//std::shared_ptr<ICNNNetwork> convertedNetwork;
//convertedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(f_ptr, originalNetwork, false);
//}