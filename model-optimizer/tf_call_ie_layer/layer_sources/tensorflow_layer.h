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
#pragma once

#include "inference_engine.hpp"
#include "ie_iextension.h"
#include <vector>

namespace IECustomExtension {

class TensorflowImplementation : public InferenceEngine::ILayerExecImpl {
public:
    explicit TensorflowImplementation(const InferenceEngine::CNNLayer *layer): _layer(*layer) {
        try {
            protobuf = _layer.GetParamAsString("protobuf");
            input_nodes_names = _layer.GetParamAsString("input_nodes_names");
            output_tensors_names = _layer.GetParamAsString("output_tensors_names");
            real_input_dims = _layer.GetParamAsString("real_input_dims");
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }


    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf, InferenceEngine::ResponseDesc *resp) noexcept override;
    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig& config, InferenceEngine::ResponseDesc *resp) noexcept override;
    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr>& inputs, std::vector<InferenceEngine::Blob::Ptr>& outputs,
                       InferenceEngine::ResponseDesc *resp) noexcept override;

    ~TensorflowImplementation() override {}

private:
    InferenceEngine::CNNLayer _layer;
    std::string protobuf;
    std::string input_nodes_names;
    std::string output_tensors_names;
    std::string real_input_dims;

    std::string errorMsg;
};

class TensorflowFactory : public InferenceEngine::ILayerImplFactory {
public:
    explicit TensorflowFactory(const InferenceEngine::CNNLayer *layer): cnnLayer(*layer) {}

    InferenceEngine::StatusCode getShapes(const std::vector<InferenceEngine::TensorDesc>& inShapes, std::vector<InferenceEngine::TensorDesc>& outShapes,
                         InferenceEngine::ResponseDesc *resp) noexcept override {
        return InferenceEngine::NOT_IMPLEMENTED;
    }

    // First implementation has more priority than next
    InferenceEngine::StatusCode getImplementations(std::vector<InferenceEngine::ILayerImpl::Ptr>& impls, InferenceEngine::ResponseDesc *resp) noexcept override {
        impls.push_back(InferenceEngine::ILayerImpl::Ptr(new TensorflowImplementation(&cnnLayer)));
        return InferenceEngine::OK;
    }

protected:
    InferenceEngine::CNNLayer cnnLayer;
};

}  // namespace IECustomExtension
