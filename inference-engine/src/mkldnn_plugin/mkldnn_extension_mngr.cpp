// Copyright (C) 2018-2019 Intel Corporation
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



