// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/configuration/options/none_layers.hpp"

namespace vpu {

inline bool skipAllLayers(const PluginConfiguration& configuration) {
    auto noneLayers = configuration.get<NoneLayersOption>();

    if (noneLayers.size() == 1) {
        const auto& val = *noneLayers.begin();
        return val == "*";
    }
    return false;
}

inline bool skipLayerType(const PluginConfiguration& configuration, const std::string& layerType) {
    return configuration.get<NoneLayersOption>().count(layerType) != 0;
}

}  // namespace vpu
