// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "multi_device_infer_request.hpp"
#include <ie_input_info.hpp>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include <blob_factory.hpp>

namespace MultiDevicePlugin {

using namespace InferenceEngine;

// ------------------------------MultiDeviceInferRequest----------------------------
MultiDeviceInferRequest::MultiDeviceInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                                 const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                                                 const InferenceEngine::SoIInferRequestInternal & request_to_share_blobs_with)
        : IInferRequestInternal(inputs, outputs),
        _requestToShareBlobsWith(request_to_share_blobs_with) {
    CreateInferRequest(request_to_share_blobs_with);
}

MultiDeviceInferRequest::MultiDeviceInferRequest(const InputsDataMap&   networkInputs,
                                                 const OutputsDataMap&  networkOutputs,
                                                 const SoIInferRequestInternal & request_to_share_blobs_with)
        : IInferRequestInternal(networkInputs, networkOutputs),
        _requestToShareBlobsWith(request_to_share_blobs_with) {
    CreateInferRequest(request_to_share_blobs_with);
}

void MultiDeviceInferRequest::CreateInferRequest(const InferenceEngine::SoIInferRequestInternal& request_to_share_blobs_with) {
    if (request_to_share_blobs_with) {
        // borrow device-friendly blobs from the request
        for (const auto &it : _networkInputs)
            _inputs[it.first] = request_to_share_blobs_with->GetBlob(it.first);
        for (const auto &it : _networkOutputs)
            _outputs[it.first] = request_to_share_blobs_with->GetBlob(it.first);
        return;
    }
    // Allocate all input blobs
    for (const auto &it : _networkInputs) {
        Layout l = it.second->getLayout();
        Precision p = it.second->getPrecision();
        SizeVector dims = it.second->getTensorDesc().getDims();

        TensorDesc desc = TensorDesc(p, dims, l);
        _inputs[it.first] = make_blob_with_precision(desc);
        _inputs[it.first]->allocate();
    }
    // Allocate all output blobs
    for (const auto &it : _networkOutputs) {
        Layout l = it.second->getLayout();
        Precision p = it.second->getPrecision();
        SizeVector dims = it.second->getTensorDesc().getDims();

        TensorDesc desc = TensorDesc(p, dims, l);
        _outputs[it.first] = make_blob_with_precision(desc);
        _outputs[it.first]->allocate();
    }
}

void MultiDeviceInferRequest::SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& blob) {
    auto exeNetwork = _exeNetwork.get();
    if (dynamic_cast<MultiDeviceExecutableNetwork*>(exeNetwork)->_networkActualNeeded
            && !dynamic_cast<MultiDeviceExecutableNetwork*>(exeNetwork)->_networkFirstReady && _requestToShareBlobsWith)
        _requestToShareBlobsWith->SetBlob(name, blob);
    else
	 IInferRequestInternal::SetBlob(name, blob);
}

InferenceEngine::Blob::Ptr MultiDeviceInferRequest::GetBlob(const std::string& name) {
    auto exeNetwork = _exeNetwork.get();
    if (dynamic_cast<MultiDeviceExecutableNetwork*>(exeNetwork)->_networkActualNeeded
            && !dynamic_cast<MultiDeviceExecutableNetwork*>(exeNetwork)->_networkFirstReady && _requestToShareBlobsWith)
        return _requestToShareBlobsWith->GetBlob(name);
    else
        return IInferRequestInternal::GetBlob(name);
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

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> MultiDeviceInferRequest::GetPerformanceCounts() const {
    IE_THROW(NotImplemented);
}

void MultiDeviceInferRequest::InferImpl() {
    IE_THROW(NotImplemented);
}

}  // namespace MultiDevicePlugin
