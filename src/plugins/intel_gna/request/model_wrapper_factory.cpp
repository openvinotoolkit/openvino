// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_wrapper_factory.hpp"

#include <memory>

#include "backend/am_intel_dnn.hpp"
#include "gna2_model_helper.hpp"

namespace GNAPluginNS {
namespace request {

std::shared_ptr<ModelWrapper> ModelWrapperFactory::create_trivial() {
    return std::make_shared<ModelWrapper>(ModelWrapper::ConstructionPassKey());
}

std::shared_ptr<ModelWrapper> ModelWrapperFactory::create_with_number_of_empty_operations(
    uint32_t number_of_operations) {
    auto wrapper = create_trivial();

    if (number_of_operations == 0) {
        return wrapper;
    }

    auto& object = wrapper->object();
    object.Operations =
        reinterpret_cast<Gna2Operation*>(gnaUserAllocator(number_of_operations * sizeof(Gna2Operation)));
    if (object.Operations == nullptr) {
        THROW_GNA_EXCEPTION << "out of memory in while allocating " << number_of_operations << " GNA layers";
    }

    object.NumberOfOperations = number_of_operations;
    for (uint32_t i = 0; i < object.NumberOfOperations; i++) {
        object.Operations[i].Type = Gna2OperationTypeNone;
        object.Operations[i].Operands = nullptr;
        object.Operations[i].NumberOfOperands = 0;
        object.Operations[i].Parameters = nullptr;
        object.Operations[i].NumberOfParameters = 0;
    }

    return wrapper;
}

std::shared_ptr<ModelWrapper> ModelWrapperFactory::create_initialized(ModelInitializer initializer) {
    auto wrapper = create_trivial();
    initializer(&wrapper->object());
    return wrapper;
}

}  // namespace request
}  // namespace GNAPluginNS
