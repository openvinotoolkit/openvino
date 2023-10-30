// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include <algorithm>

#include "extension_mngr.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

void ExtensionManager::AddExtension(const IExtensionPtr& extension) {
    _extensions.push_back(extension);
}

InferenceEngine::ILayerImpl::Ptr ExtensionManager::CreateImplementation(const std::shared_ptr<ngraph::Node>& op) {
    if (!op)
        OPENVINO_THROW("Cannot get nGraph operation!");
    for (const auto& ext : _extensions) {
        auto implTypes = ext->getImplTypes(op);
        for (const auto& type : implTypes) {
            if (type != "CPU")
                continue;
            auto impl = ext->getImplementation(op, "CPU");
            if (impl)
                return impl;
        }
    }
    return nullptr;
}

const std::vector<InferenceEngine::IExtensionPtr> & ExtensionManager::Extensions() const {
    return _extensions;
}

}   // namespace intel_cpu
}   // namespace ov
