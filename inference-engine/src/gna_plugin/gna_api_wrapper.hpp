// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gna-api-types-xnn.h>
#include "gna_plugin_log.hpp"
namespace GNAPluginNS {

/**
 * represent wrapper that capable to exception save pass c-objects
 * @tparam T
 */
template <class T>
class CPPWrapper {
};

template <>
class CPPWrapper<intel_nnet_type_t> {
 public:
    intel_nnet_type_t obj;

    CPPWrapper() {
        obj.nLayers = 0;
        obj.pLayers = nullptr;
        obj.nGroup = 0;
    }

    /**
     * creates nnet structure of n layers
     * @param n - number  of layers
     */
    explicit CPPWrapper(size_t n) {
        obj.pLayers = reinterpret_cast<intel_nnet_layer_t *>(_mm_malloc(n * sizeof(intel_nnet_layer_t), 64));
        if (obj.pLayers == nullptr) {
            THROW_GNA_EXCEPTION << "out of memory in while allocating "<< n << " GNA layers";
        }
        obj.nLayers = n;
        for (int i = 0; i < obj.nLayers; i++) {
            obj.pLayers[i].pLayerStruct = nullptr;
        }
    }
    ~CPPWrapper() {
        for (int i = 0; i < obj.nLayers; i++) {
            if (obj.pLayers[i].pLayerStruct != nullptr) {
                _mm_free(obj.pLayers[i].pLayerStruct);
            }
        }
        _mm_free(obj.pLayers);
    }
    intel_nnet_type_t * operator ->() {
        return &obj;
    }
    intel_nnet_type_t * operator *() {
        return &obj;
    }
    operator  intel_nnet_type_t &() {
        return *this;
    }
};

}  // namespace GNAPluginNS