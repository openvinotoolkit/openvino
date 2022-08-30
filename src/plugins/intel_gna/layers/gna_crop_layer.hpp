// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>
#include "gna_graph_tools.hpp"

namespace GNAPluginNS {
class GNACropLayer {
    InferenceEngine::CNNLayerPtr cropLayer;

public:
    explicit GNACropLayer(InferenceEngine::CNNLayerPtr layer) :
        cropLayer(layer)
    {}

    InferenceEngine::CNNLayerPtr getCrop() { return cropLayer; }

    /**
     * pointer to gna croped memory beginning
     */
    void *gna_ptr = nullptr;
};

/**
 * @brief returns parameters extracted from Crop layer: elements offset, elements output size and axes
 * @param cropLayer pointer to a Crop layer
 */
inline std::tuple<size_t, size_t, std::vector<int32_t>> GetCropParams(InferenceEngine::CropLayer* cropLayer) {
    IE_ASSERT(!cropLayer->axis.empty());
    IE_ASSERT(cropLayer->axis.size() == cropLayer->dim.size());
    IE_ASSERT(cropLayer->axis.size() == cropLayer->offset.size());

    std::vector<int> axis, dim, offset;
    auto inputs = cropLayer->insData.begin()->lock();
    for (int n = 0; n < cropLayer->axis.size(); n++) {
        uint32_t input_dim = GetDataDimSize(inputs, inputs->getDims().size() - cropLayer->axis[n]);
        // Exclude crop layer components that do nothing
        if (cropLayer->offset[n] == 0 && cropLayer->dim[n] == input_dim) {
            continue;
        }
        axis.push_back(cropLayer->axis[n]);
        dim.push_back(cropLayer->dim[n]);
        offset.push_back(cropLayer->offset[n]);
    }

    if (axis.size() != 1) {
        THROW_GNA_EXCEPTION <<
            "Crop layer does not support the number of (non-trivial) cropped dimensions more than 1, provided: "
            << axis.size() << ".";
    }

    size_t cropOffset = offset.front();
    size_t cropOutputSize = dim.front();

    // fix for crop on tensor dim > 2D
    for (int n = axis[0]+1; n < cropLayer->dim.size(); n++) {
        cropOffset *= cropLayer->dim[n];
        cropOutputSize *= cropLayer->dim[n];
    }

    return std::make_tuple(cropOffset, cropOutputSize, axis);
}

}  // namespace GNAPluginNS
