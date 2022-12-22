// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_wrapper.hpp"

#include "gna2_model_helper.hpp"

namespace GNAPluginNS {
namespace request {

ModelWrapper::ModelWrapper(ConstructionPassKey) {
    object_.NumberOfOperations = 0;
    object_.Operations = nullptr;
}

ModelWrapper::~ModelWrapper() {
    if (object_.NumberOfOperations > 0) {
        for (uint32_t i = 0; i < object_.NumberOfOperations; i++) {
            freeGna2Operation(object_.Operations[i]);
        }
        gnaUserFree(object_.Operations);
        object_.Operations = nullptr;
    }
}

Gna2Model& ModelWrapper::object() {
    return object_;
}

const Gna2Model& ModelWrapper::object() const {
    return object_;
}

}  // namespace request
}  // namespace GNAPluginNS
