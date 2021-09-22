// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/query_network.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/opsets/opset3.hpp"

#include <transformations/rt_info/fused_names_attribute.hpp>
#include <legacy/details/ie_cnn_network_iterator.hpp>
#include <ie_algorithm.hpp>


namespace vpu {

InferenceEngine::QueryNetworkResult getQueryNetwork(const InferenceEngine::CNNNetwork& convertedNetwork,
                                                    const std::shared_ptr<const ngraph::Function>& function,
                                                    const std::string& pluginName, const std::set<std::string>& supportedLayers) {
    InferenceEngine::QueryNetworkResult res;
    std::unordered_set<std::string> originalOps;
    for (auto& node : function->get_ops()) {
        originalOps.emplace(node->get_friendly_name());
    }

    std::unordered_set<std::string> supported;
    std::unordered_set<std::string> unsupported;

    std::unordered_set<std::string> splitNames;
    std::unordered_set<std::string> concatNames;

    ngraph::NodeVector splits;
    ngraph::NodeVector concats;

    const auto isLayerSupported = [&supportedLayers, &splitNames, &concatNames, &concats, &splits]
                                  (InferenceEngine::details::CNNNetworkIterator& layer) -> bool {
        auto node = (*layer)->getNode();
        if (std::dynamic_pointer_cast<const ::ngraph::opset3::Split>(node) != nullptr) {
            splitNames.emplace(node->get_friendly_name());
            splits.push_back(node);
            return false;
        } else if (std::dynamic_pointer_cast<const ::ngraph::opset3::Concat>(node) != nullptr) {
            concatNames.emplace(node->get_friendly_name());
            concats.push_back(node);
            return false;
        } else {
            return supportedLayers.count((*layer)->name) != 0;
        }
    };

    for (InferenceEngine::details::CNNNetworkIterator itLayer{convertedNetwork};
            itLayer != InferenceEngine::details::CNNNetworkIterator();
            itLayer++) {
        const auto fusedNode = (*itLayer)->getNode();
        if (fusedNode == nullptr) {
            continue;
        }

        for (auto& fusedLayerName : ngraph::getFusedNamesVector(fusedNode)) {
            if (InferenceEngine::details::contains(originalOps, fusedLayerName)) {
                if (isLayerSupported(itLayer)) {
                    supported.emplace(fusedLayerName);
                } else {
                    unsupported.emplace(fusedLayerName);
                }
            }
        }
    }

    for (const auto& layerName : unsupported) {
        if (supported.empty()) {
            break;
        }
        supported.erase(layerName);
    }

    unsupported.clear();
    std::function<void(std::shared_ptr<ngraph::Node>)> markParentSplitAsUnsupported = [&markParentSplitAsUnsupported, &supported, &splitNames]
                                                                                        (const std::shared_ptr<ngraph::Node>& split) {
        const auto inputs = split->inputs();
        for (const auto& input : inputs) {
            const auto& parentName = input.get_source_output().get_node()->get_friendly_name();
            if (InferenceEngine::details::contains(supported, parentName) &&
                InferenceEngine::details::contains(splitNames, parentName)) {
                markParentSplitAsUnsupported(input.get_source_output().get_node_shared_ptr());
            }
        }
        const auto& name = split->get_friendly_name();
        if (InferenceEngine::details::contains(supported, name)) {
            supported.erase(name);
        }
    };

    for (const auto& split : splits) {
        // We will mark split as a supported only if all consumers is supported
        bool is_supported = true;
        const auto outputs = split->outputs();
        for (const auto& output : outputs) {
            for (const auto& consumer : output.get_target_inputs()) {
                const auto& name = consumer.get_node()->get_friendly_name();
                if (!InferenceEngine::details::contains(supported, name) &&
                    !InferenceEngine::details::contains(concatNames, name) &&
                    !InferenceEngine::details::contains(splitNames, name)) {
                    is_supported = false;
                    break;
                }
            }
        }
        if (is_supported) {
            supported.emplace(split->get_friendly_name());
        } else {
            // If Split is not supported and it's parent is also Split, mark parent as unsupported
            markParentSplitAsUnsupported(split);
        }
    }

    for (const auto& concat : concats) {
        // We will mark concat as a supported only if all parent layers is supported
        bool is_supported = true;
        const auto inputs = concat->inputs();
        for (const auto& input : inputs) {
            const auto& name = input.get_source_output().get_node()->get_friendly_name();
            if (!InferenceEngine::details::contains(supported, name) &&
                !InferenceEngine::details::contains(concatNames, name)) {
                is_supported = false;
                break;
            }
        }
        if (is_supported) {
            supported.emplace(concat->get_friendly_name());
        }
    }

    for (const auto& node : function->get_ops()) {
        if (InferenceEngine::details::contains(supported, node->get_friendly_name())) {
            for (const auto& inputNodeOutput : node->input_values()) {
                if (ngraph::op::is_constant(inputNodeOutput.get_node()) || ngraph::op::is_parameter(inputNodeOutput.get_node())) {
                    supported.emplace(inputNodeOutput.get_node()->get_friendly_name());
                }
            }
            for (const auto& outputs : node->outputs()) {
                for (const auto& outputNodeInput : outputs.get_target_inputs()) {
                    if (ngraph::op::is_output(outputNodeInput.get_node())) {
                        supported.emplace(outputNodeInput.get_node()->get_friendly_name());
                    }
                }
            }
        }

        if (ngraph::op::is_constant(node) || ngraph::op::is_parameter(node)) {
            if (!InferenceEngine::details::contains(supported, node->output(0).get_target_inputs().begin()->get_node()->get_friendly_name())) {
                supported.erase(node->get_friendly_name());
            }
        } else if (ngraph::op::is_output(node)) {
            if (!InferenceEngine::details::contains(supported, node->input_values().begin()->get_node()->get_friendly_name())) {
                supported.erase(node->get_friendly_name());
            }
        }
    }

    for (const auto& layerName : supported) {
        res.supportedLayersMap.emplace(layerName, pluginName);
    }

    return res;
}

} // namespace vpu
