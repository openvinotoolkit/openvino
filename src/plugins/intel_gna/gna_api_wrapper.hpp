// Copyright (C) 2018-2022 Intel Corporation
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

#include <gna2-model-api.h>
#include <gna2_model_helper.hpp>

namespace GNAPluginNS {

/**
 * represent wrapper that capable to exception save pass c-objects
 * @tparam T
 */
template <class T>
class CPPWrapper {
};

template <>
class CPPWrapper<Gna2Model> {
 public:
     Gna2Model obj;

    CPPWrapper() {
        obj.NumberOfOperations = 0;
        obj.Operations = nullptr;
    }

    /**
     * creates nnet structure of n layers
     * @param n - number  of layers
     */
    explicit CPPWrapper(size_t n) {
        if (n == 0) {
            THROW_GNA_EXCEPTION << "Can't allocate array of intel_nnet_layer_t objects of zero length";
        }
        obj.Operations = reinterpret_cast<Gna2Operation*>(gnaUserAllocator(n * sizeof(Gna2Operation)));
        if (obj.Operations == nullptr) {
            THROW_GNA_EXCEPTION << "out of memory in while allocating "<< n << " GNA layers";
        }
        obj.NumberOfOperations = n;
        for (uint32_t i = 0; i < obj.NumberOfOperations; i++) {
            obj.Operations[i].Type = Gna2OperationTypeNone;
            obj.Operations[i].Operands = nullptr;
            obj.Operations[i].NumberOfOperands = 0;
            obj.Operations[i].Parameters = nullptr;
            obj.Operations[i].NumberOfParameters = 0;
        }
    }
    ~CPPWrapper() {
        if (obj.Operations != nullptr) {
            for (uint32_t i = 0; i < obj.NumberOfOperations; i++) {
                freeGna2Operation(obj.Operations[i]);
            }
            gnaUserFree(obj.Operations);
            obj.Operations = nullptr;
        }
        obj.NumberOfOperations = 0;
    }
    Gna2Model * operator ->() {
        return &obj;
    }
    Gna2Model * operator *() {
        return &obj;
    }
    operator Gna2Model &() {
        return *this;
    }
};

}  // namespace GNAPluginNS
