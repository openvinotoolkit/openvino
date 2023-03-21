// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "debug.h"
#include "legacy/ie_layers.h"

namespace ov {
namespace intel_gna {

/**
 * maps type of connection to input and output layers also stores gna_pointer for alloc request
 */
class GNAMemoryLayer {
    InferenceEngine::CNNLayerPtr inputLayer;
    InferenceEngine::CNNLayerPtr outputLayer;
    const int elementSize;

public:
    GNAMemoryLayer(InferenceEngine::CNNLayerPtr inLayer, InferenceEngine::CNNLayerPtr outLayer, int elementSize)
        : inputLayer(inLayer),
          outputLayer(outLayer),
          elementSize(elementSize) {}

    InferenceEngine::CNNLayerPtr getInput() const {
        return inputLayer;
    }
    InferenceEngine::CNNLayerPtr getOutput() const {
        return outputLayer;
    }
    InferenceEngine::SizeVector getDims() const {
        return inputLayer->outData.front()->getDims();
    }
    /**
     * @brief Get size requred for the gna memory buffer
     */
    size_t getByteSize() const {
        return InferenceEngine::details::product(getDims()) * elementSizeBytes();
    }
    /**
     * @brief Reset the gna memory
     */
    void Reset() {
        std::memset(gna_ptr, 0, reserved_size);
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
    void* gna_ptr = nullptr;
    /**
     * gna memory of this size is reserved
     */
    size_t reserved_size = 0;
    /**
     * gna memory of this offset from gna_ptr
     */
    size_t reserved_offset = 0;
    /**
     * scale factor to gna memory layer
     */
    float scale_factor = 1.0f;
};

}  // namespace intel_gna
}  // namespace ov
