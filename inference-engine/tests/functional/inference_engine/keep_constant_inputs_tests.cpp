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
#include <convert_function_to_cnn_network.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <transformations/convert_opset2_to_opset1/convert_opset2_to_opset1.hpp>
#include <transformations/convert_opset3_to_opset2/convert_opset3_to_opset2.hpp>
#include "generic_ie.hpp"
#include "graph_tools.hpp"
#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

using namespace testing;
using namespace InferenceEngine;

int isInputConstLayersInCNNNetwork(const InferenceEngine::CNNNetwork& network, std::string layerType) {
    int numberOfInputs = 0;
    InferenceEngine::CNNLayerPtr layer = nullptr;

    IE_SUPPRESS_DEPRECATED_START
    for (auto it = details::CNNNetworkIterator(network); it != details::CNNNetworkIterator(); it++) {
        layer = *it;
        if (layer->type == layerType) {
            numberOfInputs = layer->insData.size();
            std::cout << "Found layer: " << layer->name << " of type: " << layer->type << std::endl;
            std::cout << "Number of inputs: " << numberOfInputs << std::endl;
            break;
        }
    }
    IE_SUPPRESS_DEPRECATED_END
    return numberOfInputs;
}

void transformNetwork(std::shared_ptr<InferenceEngine::ICNNNetwork> clonedNetwork, bool keep_constant_inputs) {
    if (clonedNetwork->getFunction()) {
        const auto transformations_callback = [](const std::shared_ptr<const ::ngraph::Node> &node) -> bool {
            // DepthToSpace node implementation supports only equal input/output tensors with rank <= 5
            if (auto dtsOp = std::dynamic_pointer_cast<const ::ngraph::opset3::DepthToSpace>(node)) {
                return dtsOp->input_value(0).get_shape().size() <= 5lu &&
                       dtsOp->input_value(0).get_shape().size() == dtsOp->get_output_shape(0).size();
            }

            // SpaceToDepth node implementation supports only equal input/output tensors with rank <= 5
            if (auto stdOp = std::dynamic_pointer_cast<const ::ngraph::opset3::SpaceToDepth>(node)) {
                return stdOp->input_value(0).get_shape().size() <= 5lu &&
                       stdOp->input_value(0).get_shape().size() == stdOp->get_output_shape(0).size();
            }

            return std::dynamic_pointer_cast<const ::ngraph::opset2::Gelu>(node) ||
                   std::dynamic_pointer_cast<const ::ngraph::opset2::BatchToSpace>(node) ||
                   std::dynamic_pointer_cast<const ::ngraph::opset2::SpaceToBatch>(node) ||
                   std::dynamic_pointer_cast<const ::ngraph::opset3::ShuffleChannels>(node);
        };
        auto nGraphFunc = clonedNetwork->getFunction();
        // Disable shape inference (WA for generic operations)
        ::ngraph::op::GenericIE::DisableReshape noReshape(nGraphFunc);

        // Note: instead of running all Conversion Transformations you can make up your own transformation pipeline
        ngraph::pass::CommonOptimizations(transformations_callback).run_on_function(nGraphFunc);
        ngraph::pass::ConvertOpSet3ToOpSet2(transformations_callback).run_on_function(nGraphFunc);
        ngraph::pass::ConvertOpSet2ToOpSet1(transformations_callback).run_on_function(nGraphFunc);
        ngraph::pass::ConvertOpSet1ToLegacy(transformations_callback).run_on_function(nGraphFunc);
        clonedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(nGraphFunc, *clonedNetwork, keep_constant_inputs);
    }
}

TEST(KeepConstantInputsTests, ConvertConvolutionPoolReluNetworkWithTrue) {
    std::shared_ptr <ngraph::Function> f_ptr;
    f_ptr = ngraph::builder::subgraph::makeConvPoolRelu();
    InferenceEngine::CNNNetwork originalNetwork(f_ptr);
    std::shared_ptr<InferenceEngine::ICNNNetwork> clonedNetwork = InferenceEngine::cloneNetwork(originalNetwork);
    transformNetwork(clonedNetwork, true);
    InferenceEngine::CNNNetwork convertedNetwork(clonedNetwork);
    std::cout << "Check for conversion of ConvolutionPoolRelu Network with keep_constant_inputs = true" << std::endl;
    ASSERT_GT(isInputConstLayersInCNNNetwork(convertedNetwork, "Convolution"), 1);
}

TEST(KeepConstantInputsTests, ConvertConvolutionPoolReluNetworkWithFalse) {
    std::shared_ptr <ngraph::Function> f_ptr;
    f_ptr = ngraph::builder::subgraph::makeConvPoolRelu();
    InferenceEngine::CNNNetwork originalNetwork(f_ptr);
    std::shared_ptr<InferenceEngine::ICNNNetwork> clonedNetwork = InferenceEngine::cloneNetwork(originalNetwork);
    transformNetwork(clonedNetwork, false);
    InferenceEngine::CNNNetwork convertedNetwork(clonedNetwork);
    std::cout << "Check for conversion of ConvolutionPoolRelu Network with keep_constant_inputs = false" << std::endl;
    ASSERT_EQ(isInputConstLayersInCNNNetwork(convertedNetwork, "Convolution"), 1);
}

TEST(KeepConstantInputsTests, ConvertConvolutionBiasNetworkWithTrue) {
    std::shared_ptr <ngraph::Function> f_ptr;
    f_ptr = ngraph::builder::subgraph::makeConvBias();
    InferenceEngine::CNNNetwork originalNetwork(f_ptr);
    std::shared_ptr<InferenceEngine::ICNNNetwork> clonedNetwork = InferenceEngine::cloneNetwork(originalNetwork);
    transformNetwork(clonedNetwork, true);
    InferenceEngine::CNNNetwork convertedNetwork(clonedNetwork);
    std::cout << "Check for conversion of ConvolutionBias Network with keep_constant_inputs = true" << std::endl;
    ASSERT_GT(isInputConstLayersInCNNNetwork(convertedNetwork, "Convolution"), 1);
}

TEST(KeepConstantInputsTests, ConvertConvolutionBiasNetworkWithFalse) {
    std::shared_ptr <ngraph::Function> f_ptr;
    f_ptr = ngraph::builder::subgraph::makeConvBias();
    InferenceEngine::CNNNetwork originalNetwork(f_ptr);
    std::shared_ptr<InferenceEngine::ICNNNetwork> clonedNetwork = InferenceEngine::cloneNetwork(originalNetwork);
    transformNetwork(clonedNetwork, false);
    InferenceEngine::CNNNetwork convertedNetwork(clonedNetwork);
    std::cout << "Check for conversion of ConvolutionBias Network with keep_constant_inputs = false" << std::endl;
    ASSERT_EQ(isInputConstLayersInCNNNetwork(convertedNetwork, "Convolution"), 1);
}
