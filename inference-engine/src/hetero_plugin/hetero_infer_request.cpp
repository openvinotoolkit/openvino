// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hetero_infer_request.h"
#include <ie_blob.h>
#include <ie_plugin.hpp>
#include <ie_util_internal.hpp>
#include <description_buffer.hpp>
#include <debug.h>
#include <ie_layouts.h>
#include <assert.h>
#include "ie_profiling.hpp"

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
    for (auto &&ireq : _inferRequests) {
        ireq._request = ireq._network->CreateInferRequestPtr();
        // go over all inputs and get blobs from subnet infer requests
        for (auto e : ireq._oNames) {
            requestBlob(e, ireq._request);
        }
    }

    // go over all outputs and get blobs from subnet infer requests
    for (auto r : _inferRequests) {
        for (auto e : r._iNames) {
            requestBlob(e, r._request);
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
        for (auto &&ioname : desc._iNames) {
            auto iti = _inputs.find(ioname);
            if (iti != _inputs.end()) {
                auto it = _preProcData.find(ioname);
                if (it != _preProcData.end()) {
                    if (it->second.getRoiBlob() != _blobs[ioname]) {
                        r->SetBlob(ioname.c_str(), it->second.getRoiBlob());
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
        for (auto &&ioname : desc._oNames) {
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

void HeteroInferRequest::startFirstAsyncRequest() {
    auto firstAsyncRequest = _inferRequests.begin()->_request;
    firstAsyncRequest->StartAsync();
}

void HeteroInferRequest::setCallbackForLastRequest(std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>& callback) {
    auto lastRequest = _inferRequests.back()._request;
    if (lastRequest) lastRequest->SetCompletionCallback(callback);
}

void HeteroInferRequest::setCallbackSequence() {
    for (auto desc = _inferRequests.begin(); desc != _inferRequests.end(); desc++) {
        auto &currentAsyncRequest = desc->_request;
        auto nextRequestDesc = std::next(desc);
        if (nextRequestDesc != _inferRequests.end()) {
            currentAsyncRequest->SetCompletionCallback<std::function<void(InferRequest, StatusCode)>>(
                    [=](InferRequest request, StatusCode sts) {
                        IE_PROFILING_AUTO_SCOPE(Callback)
                        if (sts == OK) {
                            nextRequestDesc->_request->StartAsync();
                        }
                    });
        }
    }
}

StatusCode HeteroInferRequest::waitAllRequests(int64_t millis_timeout) {
    StatusCode status = INFER_NOT_STARTED;
    bool shareMsMode = true;
    std::chrono::high_resolution_clock::time_point startTime;
    int64_t msLeft;
    if (millis_timeout == IInferRequest::WaitMode::STATUS_ONLY ||
        millis_timeout == IInferRequest::WaitMode::RESULT_READY) {
        shareMsMode = false;
    }
    for (auto it = _inferRequests.begin(); it != _inferRequests.end(); ++it) {
        startTime = std::chrono::high_resolution_clock::now();
        status = it->_request->Wait(millis_timeout);
        msLeft = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - startTime).count();
        if (OK != status) {
            return status;
        }
        if (shareMsMode) {
            if (millis_timeout - msLeft > 0) {
                millis_timeout -= msLeft;
            } else if (it != _inferRequests.end()) {
                return RESULT_NOT_READY;
            }
        }
    }
    return status;
}
