// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <vector>
#include <string>
#include <utility>

#include <ie_common.h>
#include <legacy/ie_layers.h>
#include "dnn_types.hpp"


namespace ov {
namespace intel_gna {
namespace backend {

struct DnnComponentExtra {
    std::string name;
    intel_dnn_component_t dnnComponent;
    bool isDelayed;
    DnnComponentExtra(std::string name,
                      const intel_dnn_component_t& dnnComponent,
                      bool isDelayed) :
                      name(name), dnnComponent(dnnComponent), isDelayed(isDelayed) {}
};

/**
 * maps layer name to dnn.component, in topological order, or execution order
 */
struct DnnComponents {
    using storage_type = std::list<DnnComponentExtra>;
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

    /**
     * @brief returns Pointer to corresponding dnn layer for topology layer name
     * @return
     */
    intel_dnn_component_t* findComponent(const std::string& layerName);

    /**
     * @brief returns Pointer to const corresponding dnn layer for topology layer
     * @return
     */
    const intel_dnn_component_t* findComponent(InferenceEngine::CNNLayerPtr layer) const;

     /**
     * @brief returns const Pointer to const corresponding dnn layer name for topology layer name
     * @return
     */
    const intel_dnn_component_t* findComponent(const std::string& layerName) const;

    /**
     * @brief extract components in execution order
     */
    std::vector<intel_dnn_component_t> getExecutionOrder();

private:
    uint32_t delayedOperations = 0;
};

}  // namespace backend
}  // namespace intel_gna
}  // namespace ov
