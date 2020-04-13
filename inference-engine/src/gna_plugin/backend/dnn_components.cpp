// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <iostream>
#include <string>
#include <ie_common.h>
#include <ie_layers.h>
#include <iomanip>
#include "backend/dnn_types.h"

#include "dnn_components.hpp"

using namespace GNAPluginNS;

intel_dnn_component_t & backend::DnnComponents::addComponent(const std::string layerName, const std::string layerMetaType) {
    components.emplace_back(layerName, intel_dnn_component_t());
    auto &currentComponent = components.back().second;
#ifdef PLOT
    currentComponent.original_layer_name = components.back().first.c_str();
    std::cout << "IR layer : " << std::left << std::setw(20) << layerName << " " << layerMetaType << "_" << components.size() - 1 << std::endl;
#endif
    return currentComponent;
}

intel_dnn_component_t * backend::DnnComponents::findComponent(InferenceEngine::CNNLayerPtr __layer) {
    auto component = std::find_if(begin(components),
                                  end(components),
                                  [&](storage_type ::value_type &comp) {
                                      return comp.first == __layer->name;
                                  });
    // check for generic prev layer
    if (component != components.end()) {
        return &component->second;
    }

    return nullptr;
}
