// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "inference_engine.hpp"

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
}  // namespace GNAPluginNS
