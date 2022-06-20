// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna2_model_wrapper.hpp"

#include "gna2_model_helper.hpp"

namespace GNAPluginNS {

Gna2ModelWrapper::Gna2ModelWrapper(ConstructionPassKey) {
    object_.NumberOfOperations = 0;
    object_.Operations = nullptr;
}

Gna2ModelWrapper::~Gna2ModelWrapper() {
    if (object_.NumberOfOperations > 0) {
        for (uint32_t i = 0; i < object_.NumberOfOperations; i++) {
            freeGna2Operation(object_.Operations[i]);
        }
        gnaUserFree(object_.Operations);
        object_.Operations = nullptr;
    }
}

Gna2Model& Gna2ModelWrapper::object() {
    return object_;
}

const Gna2Model& Gna2ModelWrapper::object() const {
    return object_;
}

}  // namespace GNAPluginNS
