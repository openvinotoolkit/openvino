// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with common functions for graph transformation
 * @file graph_transformer.h
 */

#pragma once

#include <ie_icnn_network.hpp>

namespace InferenceEngine {

/**
 * @brief Replaces layer with newLayer in network
 * @param network  - graph containing the layer
 * @param layer    - layer which need to replace
 * @param newLayer - new layer instead of layer; it must have same name like a layer for replace
 */
void replaceLayerWithNewLayer(ICNNNetwork &network, const CNNLayerPtr &layer, const CNNLayerPtr &newLayer);

}  // namespace InferenceEngine
