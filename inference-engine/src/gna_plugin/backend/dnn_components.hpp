// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <string>
#include <utility>

#include <ie_common.h>

namespace GNAPluginNS {
namespace backend {
/**
 * maps layer name to dnn.component, in topological sort prev nodes will be initialized
 */
struct DnnComponents {
    using storage_type = std::list<std::pair<std::string, intel_dnn_component_t>>;
    storage_type components;
    /**
     * @brief initializes new empty intel_dnn_component_t object
     * @param layerName - layer name in IR
     * @param layerMetaType - usually either gna of original layer type
     * @return
     */
    intel_dnn_component_t & addComponent(const std::string layerName, const std::string layerMetaType);
    /**
     * @brief returns corresponding dnn layer for topology layer
     * @return
     */
    intel_dnn_component_t * findComponent(InferenceEngine::CNNLayerPtr layer);
};
}  // namespace backend
}  // namespace GNAPluginNS
