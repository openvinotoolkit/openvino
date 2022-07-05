// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>

#include "model_wrapper.hpp"

namespace GNAPluginNS {
namespace request {

class ModelWrapperFactory {
public:
    using ModelInitializer = std::function<void(Gna2Model* gnaModel)>;

    static std::shared_ptr<ModelWrapper> create_trivial();
    static std::shared_ptr<ModelWrapper> create_with_number_of_empty_operations(uint32_t number_of_operations);
    static std::shared_ptr<ModelWrapper> create_initialized(ModelInitializer initializer);
};

}  // namespace request
}  // namespace GNAPluginNS
