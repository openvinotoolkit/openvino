// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "auto_infer_request.hpp"
#include <ie_input_info.hpp>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>

namespace AutoPlugin {
    using namespace InferenceEngine;

AutoInferRequest::AutoInferRequest(const InputsDataMap&   networkInputs,
                                   const OutputsDataMap&  networkOutputs,
                                   const InferRequest&    inferRequest)
    : IInferRequestInternal(networkInputs, networkOutputs)
    , _inferRequest(inferRequest) {
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> AutoInferRequest::GetPerformanceCounts() const {
    return _inferRequest.GetPerformanceCounts();
}

void AutoInferRequest::InferImpl() {
    _inferRequest.Infer();
}

void AutoInferRequest::SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& data) {
    _inferRequest.SetBlob(name, data);
}

Blob::Ptr AutoInferRequest::GetBlob(const std::string& name) {
    return _inferRequest.GetBlob(name);
}

void AutoInferRequest::Cancel() {
    _inferRequest.Cancel();
}

}  // namespace AutoPlugin
