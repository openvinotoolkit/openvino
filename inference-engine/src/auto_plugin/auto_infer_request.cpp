// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "auto_infer_request.hpp"
#include <ie_input_info.hpp>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>

namespace AutoPlugin {
    using namespace InferenceEngine;

AutoInferRequest::AutoInferRequest(const InputsDataMap&              networkInputs,
                                   const OutputsDataMap&             networkOutputs,
                                   const SoIInferRequestInternal&    inferRequest,
                                   bool                              enablePerfCount)
    : IInferRequestInternal(networkInputs, networkOutputs)
    , _inferRequest(inferRequest)
    , _enablePerfCount(enablePerfCount) {
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> AutoInferRequest::GetPerformanceCounts() const {
    if (_enablePerfCount) {
        try {
            return _inferRequest->GetPerformanceCounts();
        } catch (...) {
            return {};
        }
    } else {
        return {};
    }
}

void AutoInferRequest::InferImpl() {
    _inferRequest->Infer();
}

void AutoInferRequest::SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& data) {
    _inferRequest->SetBlob(name, data);
}

Blob::Ptr AutoInferRequest::GetBlob(const std::string& name) {
    return _inferRequest->GetBlob(name);
}

void AutoInferRequest::Cancel() {
    _inferRequest->Cancel();
}

void AutoInferRequest::StartAsync() {
    _inferRequest->StartAsync();
}

InferenceEngine::StatusCode AutoInferRequest::Wait(int64_t millis_timeout) {
    return _inferRequest->Wait(millis_timeout);
}

void AutoInferRequest::SetCallback(Callback callback) {
    _inferRequest->SetCallback(callback);
}

}  // namespace AutoPlugin
