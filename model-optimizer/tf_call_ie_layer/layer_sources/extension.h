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
 * \brief Definition of entry point of custom library
 * \file extensibility_sample/mkldnn/sample_extension.h
 * \example extensibility_sample/mkldnn/sample_extension.h
 */
#pragma once

#include "inference_engine.hpp"
#include "ie_iextension.h"
#include <vector>
#include <string>

namespace IECustomExtension {

using namespace InferenceEngine;

class SampleExtension : public IExtension {
public:
    StatusCode getPrimitiveTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept override {
        types = new char *[1];
        size_t count = 0;
        std::string name = "TFCustomSubgraphCall";
        types[count] = new char[name.size() + 1];
        std::copy(name.begin(), name.end(), types[count]);
        types[count][name.size()] = '\0';
        return OK;
    };
    StatusCode getFactoryFor(ILayerImplFactory *&factory, const CNNLayer *cnnLayer, ResponseDesc *resp) noexcept override;
    void GetVersion(const InferenceEngine::Version *& versionInfo) const noexcept override;
    void SetLogCallback(InferenceEngine::IErrorListener &listener) noexcept override {};
    void Unload() noexcept override {};

    void Release() noexcept override {
        delete this;
    }
};

}  // namespace IECustomExtension
