// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "plugin_infer_request.hpp"
#include <ie_input_info.hpp>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include <blob_factory.hpp>

namespace PluginHelper {
using namespace InferenceEngine;

PluginInferRequest::PluginInferRequest(const InputsDataMap&              networkInputs,
                                       const OutputsDataMap&             networkOutputs,
                                       const SoIInferRequestInternal&    inferRequest)
    : IInferRequestInternal(networkInputs, networkOutputs)
    , _inferRequest(inferRequest) {
    if (inferRequest) {
        for (const auto &it : _networkInputs)
            _inputs[it.first] = _inferRequest->GetBlob(it.first);
        for (const auto &it : _networkOutputs)
            _outputs[it.first] = _inferRequest->GetBlob(it.first);
        return;
    }
    // Allocate all input blobs
    for (const auto &it : networkInputs) {
        Layout l = it.second->getLayout();
        Precision p = it.second->getPrecision();
        SizeVector dims = it.second->getTensorDesc().getDims();

        TensorDesc desc = TensorDesc(p, dims, l);
        _inputs[it.first] = make_blob_with_precision(desc);
        _inputs[it.first]->allocate();
    }
    // Allocate all output blobs
    for (const auto &it : networkOutputs) {
        Layout l = it.second->getLayout();
        Precision p = it.second->getPrecision();
        SizeVector dims = it.second->getTensorDesc().getDims();

        TensorDesc desc = TensorDesc(p, dims, l);
        _outputs[it.first] = make_blob_with_precision(desc);
        _outputs[it.first]->allocate();
    }
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> PluginInferRequest::GetPerformanceCounts() const {
    IE_THROW(NotImplemented);
}

void PluginInferRequest::InferImpl() {
    IE_THROW(NotImplemented);
}

void PluginInferRequest::SetBlobsToAnotherRequest(const SoIInferRequestInternal& req) {
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
