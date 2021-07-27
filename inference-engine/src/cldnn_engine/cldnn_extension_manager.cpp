// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include <algorithm>

#include "cldnn_extension_manager.h"
#include "ngraph/node.hpp"

using namespace CLDNNPlugin;
using namespace InferenceEngine;

void GPUExtensionManager::AddExtension(const IExtensionPtr& extension) {
    _extensions.push_back(extension);
}

bool GPUExtensionManager::IsSupportedImplType(std::string type) {
    return std::find(supportedImplTypes.begin(), supportedImplTypes.end(), type) != supportedImplTypes.end();
}

InferenceEngine::ILayerImpl::Ptr GPUExtensionManager::CreateImplementation(const std::shared_ptr<ngraph::Node>& op) {
    if (!op)
        IE_THROW() << "Invalid ngraph op is passed to extension manager";

    std::vector<ILayerImpl::Ptr> supportedImpls;
    for (const auto& ext : _extensions) {
        auto implTypes = ext->getImplTypes(op);
        for (const auto& type : implTypes) {
            if (IsSupportedImplType(type)) {
                auto impl = ext->getImplementation(op, type);
                if (impl) {
                    supportedImpls.push_back(impl);
                }
            }
        }
    }

    if (!supportedImpls.empty()) {
        // FIXME: pick the best impl
        std::cerr << "GPUExtensionManager::CreateImplementation: " << op->get_friendly_name() << std::endl;
        return supportedImpls.front();
    }

    return nullptr;
}
