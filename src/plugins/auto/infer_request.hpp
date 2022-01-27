// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <atomic>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <map>
#include <vector>
#include <utility>
#include <memory>
#include <string>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include "ie_remote_context.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {

class MultiDeviceInferRequest : public InferenceEngine::IInferRequestInternal {
public:
    using Ptr = std::shared_ptr<MultiDeviceInferRequest>;
    explicit MultiDeviceInferRequest(const InferenceEngine::InputsDataMap&  networkInputs,
                                     const InferenceEngine::OutputsDataMap& networkOutputs,
                                     const InferenceEngine::SoIInferRequestInternal & request_to_share_blobs_with,
                                     InferenceEngine::RemoteContext::Ptr ctx = nullptr);
    explicit MultiDeviceInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                     const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                                     const InferenceEngine::SoIInferRequestInternal & request_to_share_blobs_with,
                                     InferenceEngine::RemoteContext::Ptr ctx = nullptr);
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;
    void InferImpl() override;
    // Multi-Device impl specific: sets the data (blobs from the device-less requests to the specific device request)
    void SetBlobsToAnotherRequest(const InferenceEngine::SoIInferRequestInternal& req);

private:
    void CreateInferRequest(const InferenceEngine::SoIInferRequestInternal& request_to_share_blobs_with,
                            InferenceEngine::RemoteContext::Ptr ctx);
};

}  // namespace MultiDevicePlugin
