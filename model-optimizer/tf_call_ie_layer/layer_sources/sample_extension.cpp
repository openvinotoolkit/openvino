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
 * \file extensibility_sample/mkldnn/sample_extension.cpp
 * \example extensibility_sample/mkldnn/sample_extension.cpp
 */
#include "sample_extension.h"
#include "tensorflow_layer.h"
#include <string>

using namespace IECustomExtension;
using namespace InferenceEngine;
using namespace InferenceEngine::MKLDNNPlugin;

SampleExtension::SampleExtension() {
}

InferenceEngine::StatusCode SampleExtension::CreateGenericPrimitive(IMKLDNNGenericPrimitive*& primitive,
                                                                    const CNNLayerPtr& layer,
                                                                    ResponseDesc *resp) const noexcept {
    try {
        // Create layer with type
        if (layer->type.compare("TFCustomSubgraphCall") == 0) {
            primitive = new TensorflowLayer(layer);
        } else {
            if (resp != nullptr) {
                std::string errorMsg = "Unsupported layer type " + layer->type;
                errorMsg.copy(resp->msg, 255);
            }
        }
    } catch (InferenceEngine::details::InferenceEngineException ex) {
        if (resp != nullptr) {
            std::string errorMsg = ex.what();
            errorMsg.copy(resp->msg, 255);
        }
        std::cout << ex.what() << std::endl;
        return GENERAL_ERROR;
    } catch (...) {
        if (resp != nullptr) {
            std::string errorMsg = "Unknown exception.";
            errorMsg.copy(resp->msg, 255);
        }
        return GENERAL_ERROR;
    }


    return OK;
}

static Version ExtensionDescription = {
    {1, 0},             // extension API version
    "1.0",
    "Custom layer to offload computations to TensorFlow*"   // extension description message
};

void SampleExtension::GetVersion(const Version *& versionInfo) const noexcept {
    versionInfo = &ExtensionDescription;
}

// Exported function
INFERENCE_EXTENSION_API(StatusCode) CreateMKLDNNExtension(InferenceEngine::MKLDNNPlugin::IMKLDNNExtension*& ext,
                                                          ResponseDesc* resp) noexcept {
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

