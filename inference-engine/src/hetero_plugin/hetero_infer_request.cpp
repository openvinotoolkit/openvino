// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hetero_infer_request.hpp"
#include "hetero_itt.hpp"
#include <ie_blob.h>
#include <description_buffer.hpp>
#include <ie_layouts.h>
#include <ie_algorithm.hpp>
#include <cassert>
#include <map>
#include <string>

using namespace HeteroPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

HeteroInferRequest::HeteroInferRequest(InferenceEngine::InputsDataMap networkInputs,
                                       InferenceEngine::OutputsDataMap networkOutputs,
                                       const SubRequestsList& inferRequests,
                                       const std::unordered_map<std::string, std::string>& subgraphInputToOutputBlobNames) :
    IInferRequestInternal(networkInputs, networkOutputs),
    _inferRequests(inferRequests) {
    if (_networkOutputs.empty() || _networkInputs.empty()) {
        IE_THROW() << "Internal error: no information about network's output/input";
    }

    auto requestBlob([&](const std::string& blobName, InferenceEngine::SoIInferRequestInternal& r) {
        std::string intermediateBlobName = blobName;
        auto itName = subgraphInputToOutputBlobNames.find(blobName);
        if (itName != subgraphInputToOutputBlobNames.end()) {
            intermediateBlobName = itName->second;
        }
        BlobMap::iterator itBlob;
        bool emplaced = false;
        std::tie(itBlob, emplaced) = _blobs.emplace(intermediateBlobName, Blob::Ptr{});
        if (emplaced) {
            if (InferenceEngine::details::contains(_networkInputs, blobName)) {
                _subRequestFromBlobName.emplace(blobName, r._ptr.get());
                _blobs.erase(intermediateBlobName);
            } else if (InferenceEngine::details::contains(_networkOutputs, blobName)) {
                _subRequestFromBlobName.emplace(blobName, r._ptr.get());
                _blobs.erase(intermediateBlobName);
            } else {
                itBlob->second = r->GetBlob(blobName);
            }
        } else {
            r->SetBlob(blobName, itBlob->second);
        }
    });

    // go over all subnet and create requests
    for (auto&& desc : _inferRequests) {
        desc._request = {desc._network._so, desc._network->CreateInferRequest()};
        // go over all inputs and get blobs from subnet infer requests
        for (auto&& outputInfo : desc._network->GetOutputsInfo()) {
            requestBlob(outputInfo.first, desc._request);
        }
    }

    // go over all outputs and get blobs from subnet infer requests
    for (auto&& desc : _inferRequests) {
        for (auto&& inputInfo : desc._network->GetInputsInfo()) {
            requestBlob(inputInfo.first, desc._request);
        }
    }
}

void HeteroInferRequest::SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& blob) {
    auto itRequest = _subRequestFromBlobName.find(name);
    if (itRequest == _subRequestFromBlobName.end()) {
        IE_THROW() << "There is no infer requests binded to blob with name: " << name;
    }
    itRequest->second->SetBlob(name, blob);
}

InferenceEngine::Blob::Ptr HeteroInferRequest::GetBlob(const std::string& name) {
    auto itRequest = _subRequestFromBlobName.find(name);
    if (itRequest == _subRequestFromBlobName.end()) {
        IE_THROW() << "There is no infer requests binded to blob with name: " << name;
    }
    return itRequest->second->GetBlob(name);
}

void HeteroInferRequest::SetBlob(const std::string& name, const Blob::Ptr& blob, const PreProcessInfo& info) {
    auto itRequest = _subRequestFromBlobName.find(name);
    if (itRequest == _subRequestFromBlobName.end()) {
        IE_THROW() << "There is no infer requests binded to blob with name: " << name;
    }
    itRequest->second->SetBlob(name, blob, info);
}

const InferenceEngine::PreProcessInfo& HeteroInferRequest::GetPreProcess(const std::string& name) const {
    auto itRequest = _subRequestFromBlobName.find(name);
    if (itRequest == _subRequestFromBlobName.end()) {
        IE_THROW() << "There is no infer requests binded to blob with name: " << name;
    }
    return itRequest->second->GetPreProcess(name);
}

void HeteroInferRequest::InferImpl() {
    for (auto &&desc : _inferRequests) {
        OV_ITT_SCOPED_TASK(itt::domains::HeteroPlugin, desc._profilingTask);
        auto &r = desc._request;
        assert(r);
        r->Infer();
    }
}

std::map<std::string, InferenceEngineProfileInfo> HeteroInferRequest::GetPerformanceCounts() const {
    std::map<std::string, InferenceEngineProfileInfo> perfMap;
    for (size_t i = 0; i < _inferRequests.size(); i++) {
        auto perfMapRequest = _inferRequests[i]._request->GetPerformanceCounts();
        for (auto &&r : perfMapRequest) {
            perfMap[std::string("subgraph") + std::to_string(i) + ": " + r.first] = r.second;
        }
    }
    return perfMap;
}
