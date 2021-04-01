// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <legacy/ie_layers.h>

namespace GNAPluginNS {
// Split, Slice
class GNASplitLayer {
    InferenceEngine::CNNLayerPtr splitLayer;

public:
    explicit GNASplitLayer(InferenceEngine::CNNLayerPtr layer) :
        splitLayer(layer)
    {}

    InferenceEngine::CNNLayerPtr getSplit() { return splitLayer; }
    /**
     * gna memory of this size is reserved for split
     */
    size_t reserved_size = 0;
    bool output_allocation_flag = false;
    /**
     * gna memory of this offset from gna_ptr
     */
    struct SplitConnectedLayerInfo {
        SplitConnectedLayerInfo() = default;
        SplitConnectedLayerInfo(InferenceEngine::CNNLayerPtr connectedTo,
            int insDataIdx,
            size_t o,
            size_t p) :
            connectedTo(connectedTo),
            insDataIdx(insDataIdx),
            offset(o),
            pure_size(p) {}

        InferenceEngine::CNNLayerPtr  connectedTo;
        int          insDataIdx = 0;
        size_t       offset = 0;  // in number of elements of input layer
        size_t       pure_size = 0;
    };
    std::vector<SplitConnectedLayerInfo> splitOutputLayers;
};
}  // namespace GNAPluginNS
