// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "multi_device_infer_request.hpp"

namespace MultiDevicePlugin {
    using namespace InferenceEngine;
// ------------------------------MultiDeviceInferRequest----------------------------
MultiDeviceInferRequest::MultiDeviceInferRequest(const InputsDataMap&   networkInputs,
                                                 const OutputsDataMap&  networkOutputs)
        : InferRequestInternal(networkInputs, networkOutputs) {
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

void MultiDeviceInferRequest::SetBlobsToAnotherRequest(InferRequest& req) {
    for (const auto &it : _networkInputs) {
        Blob::Ptr blob;
        auto &name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        GetBlob(name.c_str(), blob);
        req.SetBlob(name.c_str(), blob);
    }
    for (const auto &it : _networkOutputs) {
        Blob::Ptr blob;
        auto &name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        GetBlob(name.c_str(), blob);
        req.SetBlob(name.c_str(), blob);
    }
}

}  // namespace MultiDevicePlugin
