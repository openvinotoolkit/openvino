// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/transformation_context.hpp"
#include <legacy/details/ie_cnn_network_iterator.hpp>
#include <legacy/details/ie_cnn_network_tools.h>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

TransformationContext::TransformationContext(ICNNNetwork& network)
    : network(network), layers(CNNNetSortTopologically(network)) {
    auto it = details::CNNNetworkIterator(&network);
    auto end = details::CNNNetworkIterator();
    while (it != end) {
        _original_precisions_map[(*it)->name] = {};
        for (auto data : (*it)->outData) _original_precisions_map[(*it)->name][data->getName()] = data->getPrecision();
        it++;
    }
}

void TransformationContext::removeLayer(const CNNLayer& layer) {
    for (size_t i = 0lu; i < layers.size(); ++i) {
        if ((layers[i] != nullptr) && (layers[i]->name == layer.name)) {
            layers[i] = nullptr;
            break;
        }
    }
}
