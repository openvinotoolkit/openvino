// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pwl_segments_creator_factory.hpp"

#include <functional>
#include <unordered_map>

#include "backend/dnn_types.h"
#include "pwl_border_values_counter_identity.hpp"
#include "pwl_segments_creator_identity.hpp"

namespace ov {
namespace intel_gna {
namespace backend {

std::shared_ptr<PWLSegmentsCreator> PWLSegmentsCreatorFactory::CreateCreator(DnnActivationType activation_type) {
    static auto create_identity = []() -> std::shared_ptr<PWLSegmentsCreator> {
        auto border_values_counter = std::make_shared<BorderValuesCounterIdentity>();
        return std::make_shared<PWLSegmentsCreatorIdentity>(border_values_counter);
    };

    std::unordered_map<DnnActivationType, std::function<std::shared_ptr<PWLSegmentsCreator>(void)>>
        supported_activations{{kActIdentity, create_identity}};

    auto activation_itr = supported_activations.find(activation_type);

    if (activation_itr == supported_activations.end()) {
        return nullptr;
    }

    return activation_itr->second();
}

}  // namespace backend
}  // namespace intel_gna
}  // namespace ov