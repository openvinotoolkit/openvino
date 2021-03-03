// Copyright (C) 2018-2020 Intel Corporation
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

#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>

namespace MultiDevicePlugin {

class MultiDeviceInferRequest : public InferenceEngine::InferRequestInternal {
public:
    using Ptr = std::shared_ptr<MultiDeviceInferRequest>;
    explicit MultiDeviceInferRequest(const InferenceEngine::InputsDataMap&  networkInputs,
                                     const InferenceEngine::OutputsDataMap& networkOutputs,
                                     InferenceEngine::InferRequest request_to_share_blobs_with);
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override {
        THROW_IE_EXCEPTION_WITH_STATUS(NOT_IMPLEMENTED);
    }
    void InferImpl() override {
        THROW_IE_EXCEPTION_WITH_STATUS(NOT_IMPLEMENTED);
    }
    // Multi-Device impl specific: sets the data (blobs from the device-less requests to the specific device request)
    void SetBlobsToAnotherRequest(InferenceEngine::InferRequest& req);
};

}  // namespace MultiDevicePlugin
