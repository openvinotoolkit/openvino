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
#include "mkldnn/mkldnn_extension.hpp"
#include <vector>
#include <string>

namespace IECustomExtension {

using namespace InferenceEngine;
using namespace InferenceEngine::MKLDNNPlugin;

class SampleExtension : public IMKLDNNExtension {
public:
    SampleExtension();

public:
    /**
     * @brief return extension version information
     * @param versionInfo pointer to version info, will be set by plugin
     */
    void GetVersion(const InferenceEngine::Version *& versionInfo) const noexcept override;

    /**
     * @brief logging is used to track what is going on inside
     * @param listener - logging sink
     */
    void SetLogCallback(InferenceEngine::IErrorListener &listener) noexcept override {}

    /**
     * @brief creates generic layer and returns a pointer to an instance
     * @param primitive - pointer to newly created layer
     * @param layer - layer parameters (source for name, type, precision, attr, weights...)
     * @param utility -  pointer to MKLDNN reorder helper*
     * @param resp - Optional: a pointer to an already allocated object to contain extra information of a failure (if occurred)
     * @return Enumeration of the resulted action: OK (0) for success.
     */
     InferenceEngine::StatusCode CreateGenericPrimitive(IMKLDNNGenericPrimitive *& primitive,
                                                        const InferenceEngine::CNNLayerPtr& layer,
                                                        ResponseDesc *resp) const noexcept override;

    /**
     * @brief could be used to cleanup resources
     */
    void Unload() noexcept override {
    }

    void Release() noexcept override {
        delete this;
    }
};

}  // namespace IECustomExtension
