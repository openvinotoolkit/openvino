// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_infer_request.hpp"

#include "gna_plugin.hpp"

namespace GNAPluginNS {

GNAInferRequest::GNAInferRequest(const std::shared_ptr<GNAPlugin>& plg,
                                 const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                 const std::vector<std::shared_ptr<const ov::Node>>& outputs)
    : InferenceEngine::IInferRequestInternal(inputs, outputs),
      plg(plg) {
    CreateInferRequest();
}

GNAInferRequest::GNAInferRequest(const std::shared_ptr<GNAPlugin>& plg,
                                 InferenceEngine::InputsDataMap networkInputs,
                                 InferenceEngine::OutputsDataMap networkOutputs)
    : InferenceEngine::IInferRequestInternal(networkInputs, networkOutputs),
      plg(plg) {
    CreateInferRequest();
}

void GNAInferRequest::InferImpl() {
    // execute input pre-processing.
    execDataPreprocessing(_inputs);
    // result returned from sync infer wait method
    auto result = plg->Infer(_inputs, _outputs);

    // if result is false we are dealing with QoS feature
    // if result is ok, next call to wait() will return Ok, if request not in gna_queue
    if (!result) {
        inferRequestIdx = -1;
    } else {
        inferRequestIdx = -2;
    }
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GNAInferRequest::GetPerformanceCounts() const {
    return plg->GetPerformanceCounts();
}

void GNAInferRequest::StartAsyncImpl() {
    // execute input pre-processing.
    execDataPreprocessing(_inputs);
    inferRequestIdx = plg->QueueInference(_inputs, _outputs);
    // workaround to unblock callback-based flows
    if (_callback) {
        auto res = Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
        std::exception_ptr exceptionPtr;
        if (res != InferenceEngine::StatusCode::OK) {
            try {
                IE_EXCEPTION_SWITCH(res,
                                    ExceptionType,
                                    InferenceEngine::details::ThrowNow<ExceptionType>{} <<=
                                    std::stringstream{}
                                    << IE_LOCATION
                                    << InferenceEngine::details::ExceptionTraits<ExceptionType>::string());
            } catch (...) {
                exceptionPtr = std::current_exception();
            }
        }
        _callback(exceptionPtr);
    }
}

InferenceEngine::StatusCode GNAInferRequest::Wait(int64_t millis_timeout) {
    if (inferRequestIdx == -1) {
        return InferenceEngine::INFER_NOT_STARTED;
    } else if (millis_timeout < -1) {
        IE_THROW(ParameterMismatch);
    }

    if (millis_timeout == InferenceEngine::InferRequest::WaitMode::RESULT_READY) {
        millis_timeout = MAX_TIMEOUT;
    }
    const auto waitStatus = plg->WaitFor(inferRequestIdx, millis_timeout);

    if (waitStatus == RequestStatus::kPending) {
        // request is still pending so Wait() is needed once again
        return InferenceEngine::RESULT_NOT_READY;
    }
    if (waitStatus == RequestStatus::kAborted) {
        // need to preserve invalid state here to avoid next Wait() from clearing it
        inferRequestIdx = -1;
        return InferenceEngine::INFER_NOT_STARTED;
    }
    return InferenceEngine::OK;
}

std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> GNAInferRequest::QueryState() {
    auto pluginStates = plg->QueryState();
    std::vector<InferenceEngine::IVariableStateInternal::Ptr> state(pluginStates.begin(), pluginStates.end());
    return plg->QueryState();
}

void GNAInferRequest::CreateInferRequest() {
    // TODO: internal connection API - better to generalize
    if (_networkOutputs.empty()) {
        THROW_GNA_EXCEPTION << "GNAInferRequest :: network has zero outputs";
    }

    // copy inputs blobs since we need to have them in separate address space to allow simultaneous infer requests
    for (auto output : _networkOutputs) {
        _outputs[output.first] = plg->GetOutputBlob(output.first, output.second->getTensorDesc().getPrecision());
    }

    for (auto input : _networkInputs) {
        _inputs[input.first] = plg->GetInputBlob(input.first, input.second->getTensorDesc().getPrecision());
    }
}

}  // namespace GNAPluginNS