// Copyright (C) 2018-2020 Intel Corporation
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
    InferRequestInternal(networkInputs, networkOutputs),
    _inferRequests(inferRequests) {
    if (_networkOutputs.empty() || _networkInputs.empty()) {
        THROW_IE_EXCEPTION << "Internal error: no information about network's output/input";
    }

    auto requestBlob([&](const std::string& blobName, InferenceEngine::InferRequest::Ptr r) {
        std::string intermediateBlobName = blobName;
        auto itName = subgraphInputToOutputBlobNames.find(blobName);
        if (itName != subgraphInputToOutputBlobNames.end()) {
            intermediateBlobName = itName->second;
        }
        BlobMap::iterator itBlob;
        bool emplaced = false;
        std::tie(itBlob, emplaced) = _blobs.emplace(intermediateBlobName, Blob::Ptr{});
        if (emplaced) {
            itBlob->second = r->GetBlob(blobName);
            if (InferenceEngine::details::contains(networkInputs, blobName)) {
                _inputs[blobName] = itBlob->second;
            } else if (InferenceEngine::details::contains(networkOutputs, blobName)) {
                _outputs[blobName] = itBlob->second;
            }
        } else {
            r->SetBlob(blobName, itBlob->second);
        }
    });

    // go over all subnet and create requests
    for (auto&& desc : _inferRequests) {
        desc._request = desc._network.CreateInferRequestPtr();
        // go over all inputs and get blobs from subnet infer requests
        for (auto&& outputInfo : desc._network.GetOutputsInfo()) {
            requestBlob(outputInfo.first, desc._request);
        }
    }

    // go over all outputs and get blobs from subnet infer requests
    for (auto&& desc : _inferRequests) {
        for (auto&& inputInfo : desc._network.GetInputsInfo()) {
            requestBlob(inputInfo.first, desc._request);
        }
    }
}

void HeteroInferRequest::SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& data) {
    InferenceEngine::InferRequestInternal::SetBlob(name, data);
    assert(!_inferRequests.empty());
    for (auto &&desc : _inferRequests) {
        auto &r = desc._request;
        assert(nullptr != r);
        InputInfo::Ptr foundInput;
        DataPtr foundOutput;
        try {
            // if `name` is input blob
            if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
                r->SetBlob(name, data, foundInput->getPreProcess());
            }
        } catch (const InferenceEngine::details::InferenceEngineException & ex) {
            std::string message = ex.what();
            if (message.find(NOT_FOUND_str) == std::string::npos)
                throw ex;
        }
    }
}

void HeteroInferRequest::InferImpl() {
    updateInOutIfNeeded();
    for (auto &&desc : _inferRequests) {
        OV_ITT_SCOPED_TASK(itt::domains::HeteroPlugin, desc._profilingTask);
        auto &r = desc._request;
        assert(nullptr != r);
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

void HeteroInferRequest::updateInOutIfNeeded() {
    OV_ITT_SCOPED_TASK(itt::domains::HeteroPlugin, "updateInOutIfNeeded");
    assert(!_inferRequests.empty());
    for (auto &&desc : _inferRequests) {
        auto &r = desc._request;
        assert(nullptr != r);
        for (auto&& inputInfo : desc._network.GetInputsInfo()) {
            auto& ioname = inputInfo.first;
            auto iti = _inputs.find(ioname);
            if (iti != _inputs.end()) {
                auto it = _preProcData.find(ioname);
                if (it != _preProcData.end()) {
                    if (it->second->getRoiBlob() != _blobs[ioname]) {
                        r->SetBlob(ioname.c_str(), it->second->getRoiBlob());
                        _blobs[ioname] = iti->second;
                    }
                } else {
                    if (iti->second != _blobs[ioname]) {
                        r->SetBlob(ioname.c_str(), iti->second);
                        _blobs[ioname] = iti->second;
                    }
                }
            }
        }
        for (auto&& outputInfo : desc._network.GetOutputsInfo()) {
            auto& ioname = outputInfo.first;
            auto ito = _outputs.find(ioname);
            if (ito != _outputs.end()) {
                if (ito->second != _blobs[ioname]) {
                    r->SetBlob(ioname.c_str(), ito->second);
                    _blobs[ioname] = ito->second;
                }
            }
        }
    }
}
