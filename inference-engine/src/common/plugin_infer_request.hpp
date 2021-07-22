// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>

namespace PluginHelper {

class PluginInferRequest : public InferenceEngine::IInferRequestInternal {
public:
    using Ptr = std::shared_ptr<PluginInferRequest>;
    explicit PluginInferRequest(const InferenceEngine::InputsDataMap&             networkInputs,
                                const InferenceEngine::OutputsDataMap&            networkOutputs,
                                const InferenceEngine::SoIInferRequestInternal&   inferRequest);
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;
    void InferImpl() override;
    void SetBlobsToAnotherRequest(const InferenceEngine::SoIInferRequestInternal& req);

private:
    InferenceEngine::SoIInferRequestInternal _inferRequest;
};

}  // namespace PluginHelper
