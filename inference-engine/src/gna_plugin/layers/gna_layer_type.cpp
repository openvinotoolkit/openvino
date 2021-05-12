// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <unordered_set>
#include <legacy/graph_tools.hpp>
#include "gna_layer_type.hpp"
#include "gna_layer_info.hpp"

GNAPluginNS::LayerType GNAPluginNS::LayerTypeFromStr(const std::string &str) {
    auto it = LayerNameToType.find(str);
    if (it != LayerNameToType.end())
        return it->second;
    else
        return NO_TYPE;
}

bool GNAPluginNS::AreLayersSupported(InferenceEngine::CNNNetwork& network, std::string& errMessage) {
    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::InputsDataMap inputs = network.getInputsInfo();
    std::unordered_set<InferenceEngine::CNNLayer *> allLayers;
    InferenceEngine::CNNLayerPtr startLayer;
    if (inputs.empty()) {
        auto outputs = network.getOutputsInfo();
        IE_ASSERT(!outputs.empty());
        // If there are no inputs start search from an output
        startLayer = getCreatorLayer(outputs.begin()->second).lock();
    } else {
        auto network_input_precision = inputs.begin()->second->getPrecision();

        if (network_input_precision != InferenceEngine::Precision::FP32 &&
            network_input_precision != InferenceEngine::Precision::I16 &&
            network_input_precision != InferenceEngine::Precision::U8) {
            errMessage = "The plugin does not support input precision with " +
                         std::string(network_input_precision.name()) +
                         " format. Supported  input precisions FP32, I16, U8\n";
            return false;
        }

        auto & secondLayers = getInputTo(inputs.begin()->second->getInputData());
        if (secondLayers.empty()) {
            errMessage = "Network consists of input layer only (GNA)\n";
            return false;
        }
        startLayer = secondLayers.begin()->second;
    }
    auto batch_size = network.getBatchSize();

    bool check_result = true;
    InferenceEngine::details::UnorderedDFS(allLayers,
                                           startLayer,
                                           [&](const InferenceEngine::CNNLayerPtr layer) {
                                               if (LayerTypeFromStr(layer->type) == LayerType::NO_TYPE) {
                                                   errMessage = "The plugin does not support layer: " + layer->name + ":" + layer->type + "\n";
                                                   check_result =  false;
                                               }
                                               if (batch_size != 1 && LayerInfo::isBatchSizeConstrained(layer->type)) {
                                                   errMessage = "topology with layer: " + layer->name + ", type: " + layer->type +
                                                                ", and batch size(" + std::to_string(batch_size) + ") != 1 not supported";
                                                   check_result =  false;
                                               }
                                           }, false);
    IE_SUPPRESS_DEPRECATED_END
    return check_result;
}
