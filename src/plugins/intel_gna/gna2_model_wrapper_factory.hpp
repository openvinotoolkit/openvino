// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>

#include "gna2_model_wrapper.hpp"

namespace GNAPluginNS {

class Gna2ModelWrapperFactory {
public:
    using ModelInitializer = std::function<void(Gna2Model* gnaModel)>;

    static std::shared_ptr<Gna2ModelWrapper> create_trivial();
    static std::shared_ptr<Gna2ModelWrapper> create_with_number_of_empty_operations(uint32_t number_of_operations);
    static std::shared_ptr<Gna2ModelWrapper> create_initialized(ModelInitializer initializer);
};

}  // namespace GNAPluginNS
