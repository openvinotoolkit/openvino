// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"
#include "request_status.hpp"

namespace GNAPluginNS {
class GNAPlugin;

class GNAInferRequest : public InferenceEngine::IInferRequestInternal {
public:
    GNAInferRequest(const std::shared_ptr<GNAPlugin>& plg,
                    const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                    const std::vector<std::shared_ptr<const ov::Node>>& outputs);
    GNAInferRequest(const std::shared_ptr<GNAPlugin>& plg,
                    InferenceEngine::InputsDataMap network_inputs,
                    InferenceEngine::OutputsDataMap network_outputs);
    /**
     * @brief Infers specified input(s) in synchronous mode
     * @note blocks all method of InferRequest while request is ongoing (running or waiting in queue)
     */
    void InferImpl() override;

    /**
     * @brief Queries performance measures per layer to get feedback of what is the most time consuming layer.
     *  Note: not all plugins may provide meaningful data
     *  @param perfMap - a map of layer names to profiling information for that layer.
     */
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

    /**
     * @brief methods with _ThreadUnsafe prefix are to implement in plugins
     * or in default wrapper (e.g. AsyncInferRequestThreadSafeDefault)
     */
    void StartAsyncImpl() override;

    InferenceEngine::StatusCode Wait(int64_t millis_timeout) override;

    std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> QueryState() override;

protected:
    bool SetRequestIndex(uint32_t request_index);
    bool IsRequestIndexValid();
    bool IsRequestCompleted();

private:
    void CreateInferRequest();
    InferenceEngine::StatusCode HandleRequestWaitStatus(const RequestStatus& request_status);
    void ValidateAndConfigureTimeout(int64_t& millis_timeout);
    void CallCleanupAndRethrowOnException(std::function<void()>&& function_to_invoke);

    static constexpr const uint32_t kRequestIndexInvalid = std::numeric_limits<uint32_t>::max();
    static constexpr const uint32_t kRequestIndexCompleted = std::numeric_limits<uint32_t>::max() - 1;

    uint32_t _infer_request_idx = kRequestIndexInvalid;
    std::shared_ptr<GNAPlugin> plg;
};
}  // namespace GNAPluginNS
