// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "auto_infer_request.hpp"
#include <ie_input_info.hpp>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>

namespace AutoPlugin {
    using namespace InferenceEngine;

AutoInferRequest::AutoInferRequest(const InputsDataMap&              networkInputs,
                                   const OutputsDataMap&             networkOutputs,
                                   const SoIInferRequestInternal&    inferRequest)
    : IInferRequestInternal(networkInputs, networkOutputs)
    , _inferRequest(inferRequest) {
    for (const auto &it : _networkInputs)
        _inputs[it.first] = _inferRequest->GetBlob(it.first);
    for (const auto &it : _networkOutputs)
        _outputs[it.first] = _inferRequest->GetBlob(it.first);
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> AutoInferRequest::GetPerformanceCounts() const {
    IE_THROW(NotImplemented);
}

void AutoInferRequest::InferImpl() {
    IE_THROW(NotImplemented);
}

void AutoInferRequest::SetBlobsToAnotherRequest(const SoIInferRequestInternal& req) {
    for (const auto &it : _networkInputs) {
        auto &name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        auto blob = GetBlob(name);
        if (req->GetBlob(name) != blob)
            req->SetBlob(name, blob);
    }
    for (const auto &it : _networkOutputs) {
        auto &name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        auto blob = GetBlob(name);
        if (req->GetBlob(name) != blob)
            req->SetBlob(name, blob);
    }
}

}  // namespace AutoPlugin
