/*
// Copyright (c) 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/
/**
 * \brief Implementation of entry point of custom library
 */
#include "extension.h"
#include "tensorflow_layer.h"
#include <string>

using namespace IECustomExtension;
using namespace InferenceEngine;

static Version ExtensionDescription = {
    {1, 0},             // extension API version
    "1.0",
    "Custom layer to offload computations to TensorFlow*"   // extension description message
};

StatusCode SampleExtension::getFactoryFor(ILayerImplFactory *&factory, const CNNLayer *cnnLayer, ResponseDesc *resp) noexcept {
    if (cnnLayer->type.compare("TFCustomSubgraphCall") == 0) 
    {
        factory = new TensorflowFactory(cnnLayer);
        return OK;
    }
    return NOT_FOUND;
}

void SampleExtension::GetVersion(const Version *& versionInfo) const noexcept {
    versionInfo = &ExtensionDescription;
}

// Exported function
INFERENCE_EXTENSION_API(StatusCode) CreateExtension(IExtension*& ext, ResponseDesc* resp) noexcept {
    try {
        ext = new SampleExtension();
        return OK;
    } catch (std::exception& ex) {
        if (resp) {
            std::string err = ((std::string)"Couldn't create extension: ") + ex.what();
            err.copy(resp->msg, 255);
        }
        return GENERAL_ERROR;
    }
}
