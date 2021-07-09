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
                                   const SoIInferRequestInternal&    inferRequest,
                                   const InferenceEngine::IExecutableNetworkInternal::Ptr autoExecutableNetwork,
                                   bool alreadyActualNetwork,
                                   bool enablePerfCount)
    : IInferRequestInternal(networkInputs, networkOutputs)
    , _inferRequest(inferRequest)
    , _autoExecutableNetwork(std::dynamic_pointer_cast<AutoPlugin::AutoExecutableNetwork>(autoExecutableNetwork))
    , _alreadyActualNetwork(alreadyActualNetwork)
    , _enablePerfCount(enablePerfCount) {
    IE_ASSERT(_autoExecutableNetwork != nullptr);
    for (const auto &it : _networkInputs)
        _inputs[it.first] = _inferRequest->GetBlob(it.first);
    for (const auto &it : _networkOutputs)
        _outputs[it.first] = _inferRequest->GetBlob(it.first);
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
    HotSwapRequests(); //safe to call here (before actual inference started)
    SetBlobsToDeviceRequest();
    _inferRequest->Infer();
}

void AutoInferRequest::SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& data) {
    IInferRequestInternal::SetBlob(name, data);
}

Blob::Ptr AutoInferRequest::GetBlob(const std::string& name) {
    return IInferRequestInternal::GetBlob(name);
}

void AutoInferRequest::Cancel() {
    _inferRequest->Cancel();
}

void AutoInferRequest::StartAsync() {
    HotSwapRequests(); //safe to call here (before actual inference started)
    SetBlobsToDeviceRequest();
    _inferRequest->StartAsync();
}

InferenceEngine::StatusCode AutoInferRequest::Wait(int64_t millis_timeout) {
    return _inferRequest->Wait(millis_timeout);
}

void AutoInferRequest::SetCallback(Callback callback) {
    _callback = callback;
    _inferRequest->SetCallback(callback);
}

void AutoInferRequest::HotSwapRequests() {
    if (!_alreadyActualNetwork) {
        InferenceEngine::SoExecutableNetworkInternal tempSoExecNetwork;
        if (_autoExecutableNetwork->TryGetActualNetwork(tempSoExecNetwork)) {
            _alreadyActualNetwork = true;
            _inferRequest = {tempSoExecNetwork, tempSoExecNetwork->CreateInferRequest()};
            _inferRequest->SetCallback(_callback);
            printf("!!! DEBUG: Hot swap happened !!!\n");
        }
    }
}

void AutoInferRequest::SetBlobsToDeviceRequest() {
        for (const auto &it : _networkInputs) {
            const auto &name = it.first;
            // this assumes the request is already in BUSY state
            auto blob = GetBlob(name);
            if (_inferRequest->GetBlob(name) != blob)
                _inferRequest->SetBlob(name, blob);
        }
        for (const auto &it : _networkOutputs) {
            const auto &name = it.first;
            // this assumes the request is already in BUSY state
            auto blob = GetBlob(name);
            if (_inferRequest->GetBlob(name) != blob)
                _inferRequest->SetBlob(name, blob);
        }
    }
}  // namespace AutoPlugin
