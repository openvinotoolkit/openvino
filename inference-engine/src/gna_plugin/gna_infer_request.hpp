// Copyright (C) 2018-2019 Intel Corporation
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
        _outputs[_networkOutputs.begin()->first] = plg->GetOutputBlob(networkOutputs.begin()->second->getPrecision());
        for (auto input : _networkInputs) {
            _inputs[input.first] =
                plg->GetInputBlob(input.first, networkInputs.begin()->second->getInputPrecision());
        }
    }
    /**
     * @brief Infers specified input(s) in synchronous mode
     * @note blocks all method of IInferRequest while request is ongoing (running or waiting in queue)
     */
    void InferImpl() override {
        // execute input pre-processing.
        execDataPreprocessing(_inputs);
        plg->Infer(_inputs, _outputs);
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
    }

    InferenceEngine::StatusCode Wait(int64_t millis_timeout) override {
        if (inferRequestIdx == -1) return InferenceEngine::INFER_NOT_STARTED;
        plg->Wait(inferRequestIdx);
        return InferenceEngine::OK;
    }
};
}  // namespace GNAPluginNS
