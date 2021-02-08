// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <unordered_set>
#include <ie_icnn_network.hpp>
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
    InferenceEngine::CNNLayerSet inputLayers;
    InferenceEngine::InputsDataMap inputs = network.getInputsInfo();
    std::unordered_set<InferenceEngine::CNNLayer *> allLayers;
    IE_ASSERT(!inputs.empty());
    auto network_input_precision = inputs.begin()->second->getPrecision();
    auto batch_size = network.getBatchSize();

    if (network_input_precision != InferenceEngine::Precision::FP32 &&
        network_input_precision != InferenceEngine::Precision::I16 &&
        network_input_precision != InferenceEngine::Precision::U8) {
        errMessage = "The plugin does not support input precision with " + std::string(network_input_precision.name()) + " format. Supported  input precisions "
                                                                                                                         "FP32, I16, U8\n";
        return false;
    }

    if (inputs.empty()) {
        errMessage = "Network is empty (GNA)\n";
        return false;
    }

    auto & secondLayers = getInputTo(inputs.begin()->second->getInputData());
    if (secondLayers.empty()) {
        errMessage = "Network consists of input layer only (GNA)\n";
        return false;
    }

    bool check_result = true;
    InferenceEngine::details::UnorderedDFS(allLayers,
                                           secondLayers.begin()->second,
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
