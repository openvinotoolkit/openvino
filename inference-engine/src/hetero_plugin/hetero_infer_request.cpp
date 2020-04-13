// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hetero_infer_request.hpp"
#include <ie_blob.h>
#include <ie_plugin.hpp>
#include <ie_util_internal.hpp>
#include <description_buffer.hpp>
#include <ie_layouts.h>
#include <cassert>
#include <map>
#include <string>

using namespace HeteroPlugin;
using namespace InferenceEngine;

HeteroInferRequest::HeteroInferRequest(InferenceEngine::InputsDataMap networkInputs,
                                       InferenceEngine::OutputsDataMap networkOutputs,
                                       const SubRequestsList &inferRequests) :
        InferRequestInternal(networkInputs, networkOutputs),
        _inferRequests(inferRequests) {
    if (_networkOutputs.empty() || _networkInputs.empty()) {
        THROW_IE_EXCEPTION << "Internal error: no information about network's output/input";
    }

    auto requestBlob([&](const std::string &e, InferenceEngine::InferRequest::Ptr r) {
        if (networkInputs.find(e) != networkInputs.end()) {
            if (_blobs.find(e) != _blobs.end()) {
                r->SetBlob(e.c_str(), _blobs[e]);
            } else {
                _blobs[e] = r->GetBlob(e.c_str());
                _inputs[e] = _blobs[e];
            }
        } else if (networkOutputs.find(e) != networkOutputs.end()) {
            if (_blobs.find(e) != _blobs.end()) {
                r->SetBlob(e.c_str(), _blobs[e]);
            } else {
                _blobs[e] = r->GetBlob(e.c_str());
                _outputs[e] = _blobs[e];
            }
        } else {
            if (_blobs.find(e) != _blobs.end()) {
                r->SetBlob(e.c_str(), _blobs[e]);
            } else {
                _blobs[e] = r->GetBlob(e.c_str());
            }
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

void HeteroInferRequest::InferImpl() {
    updateInOutIfNeeded();
    size_t i = 0;
    for (auto &&desc : _inferRequests) {
        IE_PROFILING_AUTO_SCOPE_TASK(desc._profilingTask);
        auto &r = desc._request;
        assert(nullptr != r);
        r->Infer();
    }
}

void HeteroInferRequest::GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo> &perfMap) const {
    perfMap.clear();
    for (size_t i = 0; i < _inferRequests.size(); i++) {
        auto perfMapRequest = _inferRequests[i]._request->GetPerformanceCounts();
        for (auto &&r : perfMapRequest) {
            perfMap[std::string("subgraph") + std::to_string(i) + ": " + r.first] = r.second;
        }
    }
}

void HeteroInferRequest::updateInOutIfNeeded() {
    IE_PROFILING_AUTO_SCOPE(updateInOutIfNeeded);
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
