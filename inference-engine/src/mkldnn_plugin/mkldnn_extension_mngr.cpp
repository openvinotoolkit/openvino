// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include <algorithm>

#include "mkldnn_extension_mngr.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

void MKLDNNExtensionManager::AddExtension(const IExtensionPtr& extension) {
    _extensions.push_back(extension);
}

InferenceEngine::ILayerImpl::Ptr MKLDNNExtensionManager::CreateImplementation(const std::shared_ptr<ngraph::Node>& op) {
    if (!op)
        IE_THROW() << "Cannot get nGraph operation!";
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

std::shared_ptr<InferenceEngine::ILayerImplFactory> MKLDNNExtensionManager::CreateExtensionFactory(const std::shared_ptr<ngraph::Node>& op) {
    std::shared_ptr<ILayerImplFactory> factory;
    for (auto& ext : _extensions) {
        ResponseDesc responseDesc;
        StatusCode rc = GENERAL_ERROR;
        ILayerImplFactory* factory_ptr = nullptr;
        if (auto mkldnnExt = dynamic_cast<Extensions::Cpu::MKLDNNExtensions*>(ext.get()))
            rc = mkldnnExt->getFactoryFor(factory_ptr, op, &responseDesc);
        if (rc != OK) {
            factory = nullptr;
            continue;
        } else {
            factory.reset(factory_ptr);
        }
        if (factory) {
            break;
        }
    }
    return factory;
}
