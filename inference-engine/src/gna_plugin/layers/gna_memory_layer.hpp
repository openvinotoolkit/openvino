// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "inference_engine.hpp"

namespace GNAPluginNS {
/**
* maps type of connection to input and output layers also stores gna_pointer for alloc request
*/
class GNAMemoryLayer {
    InferenceEngine::CNNLayerPtr inputLayer;
    InferenceEngine::CNNLayerPtr outputLayer;
    const int elementSize;

public:
    GNAMemoryLayer(InferenceEngine::CNNLayerPtr inLayer, InferenceEngine::CNNLayerPtr outLayer, int elementSize) :
        inputLayer(inLayer), outputLayer(outLayer), elementSize(elementSize) {
    }

    InferenceEngine::CNNLayerPtr getInput() const { return inputLayer; }
    InferenceEngine::CNNLayerPtr getOutput() const { return outputLayer; }
    InferenceEngine::SizeVector getDims() const {
        return inputLayer->outData.front()->getDims();
    }

    /**
     * @brief possible to store memory in different precision
     */
    int elementSizeBytes() const {
        return elementSize;
    }

    /**
     * pointer to gna memory request
     */
    void *gna_ptr = nullptr;
    /**
     * gna memory of this size is reserved
     */
    size_t  reserved_size = 0;
    /**
     * gna memory of this offset from gna_ptr
     */
    size_t  reserved_offset = 0;
};
}  // namespace GNAPluginNS
