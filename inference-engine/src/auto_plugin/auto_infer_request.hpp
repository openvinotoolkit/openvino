// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include <ie_blob.h>
#include <ie_common.h>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "auto_exec_network.hpp"

namespace AutoPlugin {

class AutoInferRequest : public InferenceEngine::IInferRequestInternal {
public:
    using Ptr = std::shared_ptr<AutoInferRequest>;
    explicit AutoInferRequest(const InferenceEngine::InputsDataMap&             networkInputs,
                              const InferenceEngine::OutputsDataMap&            networkOutputs,
                              const InferenceEngine::SoIInferRequestInternal&   inferRequest);
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;
    void InferImpl() override;
    void SetBlobsToAnotherRequest(const InferenceEngine::SoIInferRequestInternal& req);

private:
    InferenceEngine::SoIInferRequestInternal _inferRequest;
};

}  // namespace AutoPlugin
