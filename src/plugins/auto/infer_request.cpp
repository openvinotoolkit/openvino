// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "infer_request.hpp"
#include <ie_input_info.hpp>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include <blob_factory.hpp>

namespace MultiDevicePlugin {

using namespace InferenceEngine;

// ------------------------------MultiDeviceInferRequest----------------------------
MultiDeviceInferRequest::MultiDeviceInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                                 const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                                                 const InferenceEngine::SoIInferRequestInternal & request_to_share_blobs_with,
                                                 InferenceEngine::RemoteContext::Ptr ctx)
        : IInferRequestInternal(inputs, outputs),
          _sharedRequest(request_to_share_blobs_with)  {
    CreateInferRequest(request_to_share_blobs_with, ctx);
}

MultiDeviceInferRequest::MultiDeviceInferRequest(const InputsDataMap&   networkInputs,
                                                 const OutputsDataMap&  networkOutputs,
                                                 const SoIInferRequestInternal & request_to_share_blobs_with,
                                                 InferenceEngine::RemoteContext::Ptr ctx)
        : IInferRequestInternal(networkInputs, networkOutputs),
          _sharedRequest(request_to_share_blobs_with) {
    CreateInferRequest(request_to_share_blobs_with, ctx);
}

void MultiDeviceInferRequest::CreateInferRequest(const InferenceEngine::SoIInferRequestInternal& request_to_share_blobs_with,
            InferenceEngine::RemoteContext::Ptr ctx) {
    if (request_to_share_blobs_with) {
        // do not need to touch multi memory blobs
        return;
    }
    // Allocate all input blobs
    for (const auto &it : _networkInputs) {
        auto l = it.second->getLayout();
        auto p = it.second->getPrecision();
        auto dims = it.second->getTensorDesc().getDims();

        TensorDesc desc = TensorDesc(p, dims, l);
        if (ctx) {
            _inputs[it.first] = ctx->CreateHostBlob(desc);
        } else {
            _inputs[it.first] = make_blob_with_precision(desc);
        }
        _inputs[it.first]->allocate();
    }
    // Allocate all output blobs
    for (const auto &it : _networkOutputs) {
        auto l = it.second->getLayout();
        auto p = it.second->getPrecision();
        auto dims = it.second->getTensorDesc().getDims();

        TensorDesc desc = TensorDesc(p, dims, l);
        if (ctx) {
            _outputs[it.first] = ctx->CreateHostBlob(desc);
        } else {
            _outputs[it.first] = make_blob_with_precision(desc);
        }
        _outputs[it.first]->allocate();
    }
}
void MultiDeviceInferRequest::SetBlobsToAnotherRequest(const SoIInferRequestInternal& req) {
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

void MultiDeviceInferRequest::SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& blob) {
    if (_sharedRequest)
        _sharedRequest->SetBlob(name, blob);
    else
        IInferRequestInternal::SetBlob(name, blob);
}

IE_SUPPRESS_DEPRECATED_START
void MultiDeviceInferRequest::SetBlob(const std::string& name, const Blob::Ptr& blob, const PreProcessInfo& info) {
    if (_sharedRequest)
        _sharedRequest->SetBlob(name, blob, info);
    else
        IInferRequestInternal::SetBlob(name, blob, info);
}
IE_SUPPRESS_DEPRECATED_END

InferenceEngine::Blob::Ptr MultiDeviceInferRequest::GetBlob(const std::string& name) {
    if (_sharedRequest)
        return _sharedRequest->GetBlob(name);
    else
        return IInferRequestInternal::GetBlob(name);
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> MultiDeviceInferRequest::GetPerformanceCounts() const {
    if (_sharedRequest) {
        return _sharedRequest->GetPerformanceCounts();
    } else {
        // get the profiling info directly from target infer request
        // not thread-safe for plugin like GPU, see CVS-86034
        if (_scheduledRequest)
            return _scheduledRequest->GetPerformanceCounts();
        else
            IE_THROW() << "Performance counters were not enabled";
    }
}

std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> MultiDeviceInferRequest::QueryState() {
    if (_sharedRequest)
        return _sharedRequest->QueryState();
    IE_THROW(NotImplemented);
}

}  // namespace MultiDevicePlugin
