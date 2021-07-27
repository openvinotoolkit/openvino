// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_extensions.h"
#include "ngraph/node.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Gpu {

std::vector<std::string> GPUExtensions::getImplTypes(const std::shared_ptr<ngraph::Node>& node) {
    if (extensionLayers.find(node->get_type_name()) != extensionLayers.end()) {
        std::cerr << "FOUND CUSTOM IMPL FOR: " << node->get_friendly_name() << " (" << node->get_type_name() << ")" << std::endl;
        return {"OCL"};
    } else {
        return {};
    }
}

ILayerImpl::Ptr GPUExtensions::getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) {
    std::vector<decltype(extensionLayers)::value_type> matches;
    std::copy_if(extensionLayers.begin(), extensionLayers.end(), std::back_inserter(matches),
    [&](const std::pair<std::string, FactoryType>& e) {
        if (e.first != node->get_type_name())
            return false;

        return true;
    });

    if (matches.empty())
        return nullptr;

    return matches.front().second(node);
}

}  // namespace Gpu
}  // namespace Extensions
}  // namespace InferenceEngine
