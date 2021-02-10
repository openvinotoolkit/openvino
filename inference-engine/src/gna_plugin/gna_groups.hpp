// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/graph_tools.hpp>
#include "gna_plugin_log.hpp"
#include "layers/gna_layer_info.hpp"

namespace GNAPluginNS {
/**
 * @brief returns a pointer to 2D reshaped data to satisfy maximum size of zero dimension
 * @param input a pointer to data to be reshaped
 * @param maxZeroDimSize the maximum size of zero dimension
 */
inline InferenceEngine::DataPtr Get2DReshapedData(InferenceEngine::DataPtr input, size_t maxZeroDimSize) {
    auto dims = input->getDims();
    uint32_t numRowsIn = InferenceEngine::details::product(begin(dims), end(dims));
    uint32_t numColumnsIn = 1;
    // Rows number should be 8-elements aligned
    if (numRowsIn % 8 == 0) {
        if (dims.size() >= 2 || dims[0] >= maxZeroDimSize) {
            size_t indexDivide = maxZeroDimSize;
            while (indexDivide > 1) {
                if ((numRowsIn / 8) % indexDivide == 0) break;
                --indexDivide;
            }
            numRowsIn /= indexDivide;
            numColumnsIn = indexDivide;
        }
    }

    InferenceEngine::SizeVector newDims(dims.size(), 1);
    newDims[0] = numColumnsIn;
    newDims[1] = numRowsIn;
    return std::make_shared<InferenceEngine::Data>(input->getName(),
        InferenceEngine::TensorDesc(input->getPrecision(), newDims, input->getLayout()));
}

/**
 * @brief returns true if input data should be 2D reshaped for the layer
 * @param layer
 */
inline bool HasTo2DReshapeData(InferenceEngine::CNNLayerPtr layer) {
    if (GNAPluginNS::LayerInfo(layer).isPower())
        return true;

    if (!GNAPluginNS::LayerInfo(layer).isScaleShift())
        return false;

    // Don't reshape user-defined ScaleShift layers
    if (layer->name.rfind("SyntheticScaleShift", 0) == std::string::npos)
        return false;

    // Don't reshape diagonallayers with bias connection
    return !GNAPluginNS::LayerInfo(getCreatorLayer(layer->insData.front().lock()).lock()).has32BOutput();
}
} // namespace GNAPluginNS