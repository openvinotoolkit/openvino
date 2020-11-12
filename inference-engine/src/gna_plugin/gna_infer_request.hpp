// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <map>

#include "cpp_interfaces/impl/ie_infer_request_internal.hpp"
#include "gna_plugin.hpp"

namespace GNAPluginNS {

class GNAInferRequest : public InferenceEngine::AsyncInferRequestInternal {
    std::shared_ptr<GNAPlugin> plg;
    uint32_t inferRequestIdx = -1;

 public:
    GNAInferRequest(const std::shared_ptr<GNAPlugin>& plg,
                    InferenceEngine::InputsDataMap networkInputs,
                    InferenceEngine::OutputsDataMap networkOutputs)
        : InferenceEngine::AsyncInferRequestInternal(networkInputs, networkOutputs), plg(plg) {
        // TODO: internal connection API - better to generalize
        if (networkOutputs.empty()) {
            THROW_GNA_EXCEPTION << "GNAInferRequest :: network has zero outputs";
        }
        if (networkInputs.empty()) {
            THROW_GNA_EXCEPTION << "GNAInferRequest :: network has zero inputs";
        }

        // copy inputs blobs since we need to have them in separate address space to allow simultaneous infer requests
        for (auto output : _networkOutputs) {
            _outputs[output.first] =
                plg->GetOutputBlob(output.first, output.second->getTensorDesc().getPrecision());
        }

        for (auto input : _networkInputs) {
            _inputs[input.first] =
                plg->GetInputBlob(input.first, input.second->getTensorDesc().getPrecision());
        }
    }
    /**
     * @brief Infers specified input(s) in synchronous mode
     * @note blocks all method of IInferRequest while request is ongoing (running or waiting in queue)
     */
    void InferImpl() override {
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

    /**
     * @brief Queries performance measures per layer to get feedback of what is the most time consuming layer.
     *  Note: not all plugins may provide meaningful data
     *  @param perfMap - a map of layer names to profiling information for that layer.
     */
    void GetPerformanceCounts(std::map<std::string,
                                               InferenceEngine::InferenceEngineProfileInfo> &perfMap) const override {
        plg->GetPerformanceCounts(perfMap);
    }

    /**
     * @brief methods with _ThreadUnsafe prefix are to implement in plugins
     * or in default wrapper (e.g. AsyncInferRequestThreadSafeDefault)
     */
    void StartAsyncImpl() override {
        // execute input pre-processing.
        execDataPreprocessing(_inputs);
        inferRequestIdx = plg->QueueInference(_inputs, _outputs);
        // workaround to unblock callback-based flows
        if (_callback) {
            auto infer_request = _publicInterface.lock();
            IE_ASSERT(infer_request != nullptr);
            auto res = Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
            _callback(infer_request, res);
        }
    }


    InferenceEngine::StatusCode Wait(int64_t millis_timeout) override {
        if (inferRequestIdx == -1) {
            return InferenceEngine::INFER_NOT_STARTED;
        } else if (millis_timeout < -1) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str;
        }

        if (millis_timeout == InferenceEngine::IInferRequest::WaitMode::RESULT_READY) {
            millis_timeout = MAX_TIMEOUT;
        }
        const auto waitStatus = plg->WaitFor(inferRequestIdx, millis_timeout);

        if (waitStatus == GNA_REQUEST_PENDING) {
            // request is still pending so Wait() is needed once again
            return InferenceEngine::RESULT_NOT_READY;
        }
        if (waitStatus == GNA_REQUEST_ABORTED) {
            // need to preserve invalid state here to avoid next Wait() from clearing it
            inferRequestIdx = -1;
            return InferenceEngine::INFER_NOT_STARTED;
        }
        return InferenceEngine::OK;
    }

    IE_SUPPRESS_DEPRECATED_START
    std::vector<InferenceEngine::IVariableStateInternal::Ptr>  QueryState() override {
        auto pluginStates = plg->QueryState();
        std::vector<InferenceEngine::IVariableStateInternal::Ptr> state(pluginStates.begin(), pluginStates.end());
        return plg->QueryState();
    }
    IE_SUPPRESS_DEPRECATED_END
};
}  // namespace GNAPluginNS
