// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include <w_unistd.h>
#include <w_dirent.h>
#include <debug.h>
#include <algorithm>
#include <file_utils.h>

#include "mkldnn_extension_mngr.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::MKLDNNPlugin;

IMKLDNNGenericPrimitive* MKLDNNExtensionManager::CreateExtensionPrimitive(const CNNLayerPtr& layer) {
    IMKLDNNGenericPrimitive* primitive = nullptr;

    // last registered has a priority
    for (auto ext = _extensions.rbegin(); ext != _extensions.rend(); ++ext) {
        ResponseDesc respDesc;
        StatusCode rc;
        auto *mkldnnExtension = dynamic_cast<IMKLDNNExtension *>(ext->get());
        if (mkldnnExtension != nullptr) {
            // If extension does not want to provide impl it should just return OK and do nothing.
            rc = mkldnnExtension->CreateGenericPrimitive(primitive, layer, &respDesc);
            if (rc != OK) {
                primitive = nullptr;
                continue;
            }

            if (primitive != nullptr) {
                break;
            }
        }
    }
    return primitive;
}

void MKLDNNExtensionManager::AddExtension(IExtensionPtr extension) {
    _extensions.push_back(extension);
}

InferenceEngine::ILayerImplFactory* MKLDNNExtensionManager::CreateExtensionFactory(
        const InferenceEngine::CNNLayerPtr &layer) {
    if (!layer)
        THROW_IE_EXCEPTION << "Cannot get cnn layer!";
    ILayerImplFactory* factory = nullptr;
    for (auto& ext : _extensions) {
        ResponseDesc responseDesc;
        StatusCode rc;
        rc = ext->getFactoryFor(factory, layer.get(), &responseDesc);
        if (rc != OK) {
            factory = nullptr;
            continue;
        }
        if (factory != nullptr) {
            break;
        }
    }
    return factory;
}



