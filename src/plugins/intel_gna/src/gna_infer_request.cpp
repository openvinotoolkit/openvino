// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_infer_request.hpp"

#include "gna_plugin.hpp"

namespace ov {
namespace intel_gna {

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

    auto infer_call = [&]() {
        auto result = plg->Infer(_inputs, _outputs);
        // if result is false we are dealing with QoS feature and set kRequestIndexInvalid
        // if result is ok we set kRequestIndexCompleted to not execute request if it is not
        // in the queue.
        auto result_request_index = result ? kRequestIndexCompleted : kRequestIndexInvalid;
        SetRequestIndex(result_request_index);
    };

    CallCleanupAndRethrowOnException(std::move(infer_call));
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GNAInferRequest::GetPerformanceCounts() const {
    return plg->GetPerformanceCounts();
}

void GNAInferRequest::StartAsyncImpl() {
    // execute input pre-processing.
    execDataPreprocessing(_inputs);

    auto queue_call = [&]() {
        SetRequestIndex(plg->QueueInference(_inputs, _outputs));
    };

    CallCleanupAndRethrowOnException(std::move(queue_call));

    // workaround to unblock callback-based flows
    if (_callback) {
        auto res = Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
        std::exception_ptr exceptionPtr;
        if (res != InferenceEngine::StatusCode::OK) {
            try {
                IE_EXCEPTION_SWITCH(
                    res,
                    ExceptionType,
                    InferenceEngine::details::ThrowNow<ExceptionType>{IE_LOCATION_PARAM} <<= std::stringstream{});
            } catch (...) {
                exceptionPtr = std::current_exception();
            }
        }
        _callback(exceptionPtr);
    }
}

InferenceEngine::StatusCode GNAInferRequest::Wait(int64_t millis_timeout) {
    if (!IsRequestIndexValid()) {
        return InferenceEngine::INFER_NOT_STARTED;
    }

    ValidateAndConfigureTimeout(millis_timeout);

    if (IsRequestCompleted()) {
        return InferenceEngine::OK;
    }

    auto waitStatus = RequestStatus::kAborted;
    auto wait_call = [&]() {
        waitStatus = plg->WaitFor(_infer_request_idx, millis_timeout);
    };
    CallCleanupAndRethrowOnException(std::move(wait_call));

    return HandleRequestWaitStatus(waitStatus);
}

std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> GNAInferRequest::QueryState() {
    auto pluginStates = plg->QueryState();
    std::vector<InferenceEngine::IVariableStateInternal::Ptr> state(pluginStates.begin(), pluginStates.end());
    return plg->QueryState();
}

bool GNAInferRequest::IsRequestIndexValid() {
    return _infer_request_idx != kRequestIndexInvalid;
}

bool GNAInferRequest::IsRequestCompleted() {
    return _infer_request_idx == kRequestIndexCompleted;
}

bool GNAInferRequest::SetRequestIndex(uint32_t request_index) {
    return (_infer_request_idx = request_index);
}

void GNAInferRequest::ValidateAndConfigureTimeout(int64_t& millis_timeout) {
    if (millis_timeout == InferenceEngine::InferRequest::WaitMode::RESULT_READY) {
        millis_timeout = MAX_TIMEOUT;
    }

    if (millis_timeout < 0) {
        IE_THROW(ParameterMismatch) << "Invalid timeout value in milliseconds: " << millis_timeout << "!";
    }
}

InferenceEngine::StatusCode GNAInferRequest::HandleRequestWaitStatus(const RequestStatus& request_status) {
    if (request_status == RequestStatus::kPending) {
        // request is still pending so Wait() is needed once again
        return InferenceEngine::RESULT_NOT_READY;
    }

    if (request_status == RequestStatus::kAborted) {
        // need to preserve invalid state here to avoid next Wait() from clearing it
        SetRequestIndex(kRequestIndexInvalid);
        return InferenceEngine::INFER_NOT_STARTED;
    }

    if (request_status == RequestStatus::kCompletedWithError) {
        SetRequestIndex(kRequestIndexInvalid);
        THROW_GNA_EXCEPTION << "Error when waiting for inference results!";
    }

    return InferenceEngine::OK;
}

void GNAInferRequest::CallCleanupAndRethrowOnException(std::function<void()>&& function_to_invoke) {
    try {
        function_to_invoke();
    } catch (...) {
        // need to preserve invalid state here to avoid next Wait() from clearing it
        // and next rethrow issue.
        SetRequestIndex(kRequestIndexInvalid);
        throw;
    }
}

void GNAInferRequest::CreateInferRequest() {
    // TODO: internal connection API - better to generalize
    if (_networkOutputs.empty()) {
        THROW_GNA_EXCEPTION << "GNAInferRequest :: network has zero outputs";
    }

    // copy inputs blobs since we need to have them in separate address space to allow simultaneous infer requests
    for (const auto& output : _networkOutputs) {
        _outputs[output.first] = plg->GetOutputBlob(output.first, output.second->getTensorDesc().getPrecision());
    }

    for (const auto& input : _networkInputs) {
        _inputs[input.first] = plg->GetInputBlob(input.first, input.second->getTensorDesc().getPrecision());
    }
}

}  // namespace intel_gna
}  // namespace ov
