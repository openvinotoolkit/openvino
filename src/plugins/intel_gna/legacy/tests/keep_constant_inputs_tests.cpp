// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cpp/ie_cnn_network.h>
#include <gtest/gtest.h>

#include <functional_test_utils/precision_utils.hpp>
#include <ie_precision.hpp>
#include <legacy/cnn_network_impl.hpp>  // deprecated API
#include <legacy/convert_function_to_cnn_network.hpp>
#include <legacy/details/ie_cnn_network_iterator.hpp>  // deprecated API
#include <legacy/ngraph_ops/convolution_ie.hpp>
#include <legacy/ngraph_ops/fully_connected.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/opset_conversions/convert_opset2_to_opset1.hpp>
#include <transformations/opset_conversions/convert_opset3_to_opset2.hpp>

#include "ov_models/subgraph_builders.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

using namespace testing;
using namespace InferenceEngine;

int numberOfInputsForLayerInCNNNetwork(const InferenceEngine::CNNNetwork& network, std::string layerType) {
    int numberOfInputs = 0;

    IE_SUPPRESS_DEPRECATED_START
    for (auto it = details::CNNNetworkIterator(network); it != details::CNNNetworkIterator(); it++) {
        InferenceEngine::CNNLayerPtr layer = *it;
        if (layer->type == layerType) {
            numberOfInputs = (int)layer->insData.size();
            break;
        }
    }
    IE_SUPPRESS_DEPRECATED_END
    return numberOfInputs;
}

void transformNetwork(InferenceEngine::CNNNetwork& clonedNetwork, bool keep_constant_inputs) {
    if (clonedNetwork.getFunction()) {
        auto nGraphFunc = clonedNetwork.getFunction();
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::CommonOptimizations>();
        manager.register_pass<ov::pass::ConvertOpSet3ToOpSet2>();
        manager.register_pass<ov::pass::ConvertOpSet2ToOpSet1>();
        manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
        manager.run_passes(nGraphFunc);
        IE_SUPPRESS_DEPRECATED_START
        clonedNetwork = InferenceEngine::CNNNetwork(
            InferenceEngine::details::convertFunctionToICNNNetwork(nGraphFunc, clonedNetwork, keep_constant_inputs));
        IE_SUPPRESS_DEPRECATED_END
    }
}

TEST(KeepConstantInputsTests, ConvertConvolutionPoolReluNetworkWithTrue) {
    std::shared_ptr<ov::Model> f_ptr;
    f_ptr = ngraph::builder::subgraph::makeConvPoolRelu();
    InferenceEngine::CNNNetwork network(f_ptr), originalNetwork = network;
    transformNetwork(originalNetwork, true);
    ASSERT_EQ(numberOfInputsForLayerInCNNNetwork(originalNetwork, "Convolution"), 2);
}

TEST(KeepConstantInputsTests, ConvertConvolutionPoolReluNetworkWithFalse) {
    std::shared_ptr<ov::Model> f_ptr;
    f_ptr = ngraph::builder::subgraph::makeConvPoolRelu();
    InferenceEngine::CNNNetwork network(f_ptr), originalNetwork = network;
    transformNetwork(originalNetwork, false);
    ASSERT_EQ(numberOfInputsForLayerInCNNNetwork(originalNetwork, "Convolution"), 1);
}

TEST(KeepConstantInputsTests, ConvertConvolutionBiasNetworkWithTrue) {
    std::shared_ptr<ov::Model> f_ptr;
    f_ptr = ngraph::builder::subgraph::makeConvBias();
    InferenceEngine::CNNNetwork network(f_ptr), originalNetwork = network;
    transformNetwork(originalNetwork, true);
    ASSERT_EQ(numberOfInputsForLayerInCNNNetwork(originalNetwork, "Convolution"), 3);
}

TEST(KeepConstantInputsTests, ConvertConvolutionBiasNetworkWithFalse) {
    std::shared_ptr<ov::Model> f_ptr;
    f_ptr = ngraph::builder::subgraph::makeConvBias();
    InferenceEngine::CNNNetwork network(f_ptr), originalNetwork = network;
    transformNetwork(originalNetwork, false);
    ASSERT_EQ(numberOfInputsForLayerInCNNNetwork(originalNetwork, "Convolution"), 1);
}

TEST(KeepConstantInputsTests, ConvertFullyConnectedNetworkWithTrue) {
    std::shared_ptr<ov::Model> f_ptr;
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 128});
    auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786, 128}, {1});
    auto empty_bias = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786}, {0});
    auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, empty_bias, ov::Shape{1, 786});
    f_ptr = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});
    InferenceEngine::CNNNetwork network(f_ptr), originalNetwork = network;
    transformNetwork(originalNetwork, true);
    ASSERT_EQ(numberOfInputsForLayerInCNNNetwork(originalNetwork, "FullyConnected"), 3);
}

TEST(KeepConstantInputsTests, ConvertFullyConnectedNetworkWithFalse) {
    std::shared_ptr<ov::Model> f_ptr;
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 128});
    auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786, 128}, {1});
    auto empty_bias = ov::opset1::Constant::create(ov::element::f32, ov::Shape{786}, {0});
    auto fc = std::make_shared<ngraph::op::FullyConnected>(input1, weights, empty_bias, ov::Shape{1, 786});
    f_ptr = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});
    InferenceEngine::CNNNetwork network(f_ptr), originalNetwork = network;
    transformNetwork(originalNetwork, false);
    ASSERT_EQ(numberOfInputsForLayerInCNNNetwork(originalNetwork, "FullyConnected"), 1);
}
