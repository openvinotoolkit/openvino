// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>
#include <gna_plugin_log.hpp>

namespace GNAPluginNS {
class GNACropLayer {
    InferenceEngine::CNNLayerPtr cropLayer;

public:
    explicit GNACropLayer(InferenceEngine::CNNLayerPtr layer) :
        cropLayer(layer) {
        IE_ASSERT(layer != nullptr);
    }
    void * getGnaPtr() const {
        gnalog() << "[GNACropLayer: " << cropLayer->name << "] getGnaPtr() == " << gna_ptr << "\n";
        return gna_ptr;
    }
    void setGnaPtr(void* ptr) {
        IE_ASSERT(ptr);
        IE_ASSERT(gna_ptr == nullptr);
        gna_ptr = ptr;
        gnalog() << "[GNACropLayer: " << cropLayer->name << "] setGnaPtr(" << ptr << ")\n";
    }

private:
    InferenceEngine::CNNLayerPtr getCrop() { return cropLayer; }

public:
    /**
     * pointer to gna croped memory beginning
     */
    void *gna_ptr = nullptr;
};
}  // namespace GNAPluginNS
