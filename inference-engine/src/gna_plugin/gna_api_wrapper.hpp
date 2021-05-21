// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined __INTEL_COMPILER || defined _MSC_VER
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif
#include "backend/gna_types.h"
#include "gna_plugin_log.hpp"

#if GNA_LIB_VER == 2
#include <gna2-model-api.h>
#include <gna2_model_helper.hpp>
#endif

namespace GNAPluginNS {

/**
 * represent wrapper that capable to exception save pass c-objects
 * @tparam T
 */
template <class T>
class CPPWrapper {
};

#if GNA_LIB_VER == 2
struct Gna2ModelWithMeta {
    Gna2Model gnaModel;
    std::vector<std::string> gnaModelMeta;
};

template <>
class CPPWrapper<Gna2Model> {
 public:
     Gna2ModelWithMeta obj;

    CPPWrapper() {
        obj.gnaModel.NumberOfOperations = 0;
        obj.gnaModel.Operations = nullptr;
    }

    /**
     * creates nnet structure of n layers
     * @param n - number  of layers
     */
    explicit CPPWrapper(size_t n) {
        if (n == 0) {
            THROW_GNA_EXCEPTION << "Can't allocate array of intel_nnet_layer_t objects of zero length";
        }
        obj.gnaModel.Operations = reinterpret_cast<Gna2Operation*>(gnaUserAllocator(n * sizeof(Gna2Operation)));
        if (obj.gnaModel.Operations == nullptr) {
            THROW_GNA_EXCEPTION << "out of memory in while allocating "<< n << " GNA layers";
        }
        obj.gnaModel.NumberOfOperations = n;
        for (uint32_t i = 0; i < obj.gnaModel.NumberOfOperations; i++) {
            obj.gnaModel.Operations[i].Type = Gna2OperationTypeNone;
            obj.gnaModel.Operations[i].Operands = nullptr;
            obj.gnaModel.Operations[i].NumberOfOperands = 0;
            obj.gnaModel.Operations[i].Parameters = nullptr;
            obj.gnaModel.Operations[i].NumberOfParameters = 0;
        }
    }
    ~CPPWrapper() {
        if (obj.gnaModel.Operations != nullptr) {
            for (uint32_t i = 0; i < obj.gnaModel.NumberOfOperations; i++) {
                freeGna2Operation(obj.gnaModel.Operations[i]);
            }
            gnaUserFree(obj.gnaModel.Operations);
            obj.gnaModel.Operations = nullptr;
        }
        obj.gnaModel.NumberOfOperations = 0;
    }
    Gna2ModelWithMeta* operator ->() {
        return &obj;
    }
    Gna2ModelWithMeta* operator *() {
        return &obj;
    }
    operator Gna2ModelWithMeta&() {
        return *this;
    }
};
#else
template <>
class CPPWrapper<gna_nnet_type_t> {
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
        if (n == 0) {
            THROW_GNA_EXCEPTION << "Can't allocate array of intel_nnet_layer_t objects of zero length";
        }
        obj.pLayers = reinterpret_cast<intel_nnet_layer_t *>(_mm_malloc(n * sizeof(intel_nnet_layer_t), 64));
        if (obj.pLayers == nullptr) {
            THROW_GNA_EXCEPTION << "out of memory in while allocating " << n << " GNA layers";
        }
        obj.nLayers = n;
        for (int i = 0; i < obj.nLayers; i++) {
            obj.pLayers[i].pLayerStruct = nullptr;
        }
        obj.nGroup = 0;
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
    operator intel_nnet_type_t &() {
        return *this;
    }
};
#endif

}  // namespace GNAPluginNS
