// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <mutex>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include <unordered_map>

#include "ie_icore.hpp"
#include "auto_request_wrapper.hpp"
#include "async_infer_request.hpp"
#include "plugin.hpp"

#include "utils/log_util.hpp"

// ------------------------------MultiDeviceExecutableNetwork----------------------------
namespace MultiDevicePlugin {
using namespace InferenceEngine;

InferenceEngine::IInferRequestInternal::Ptr AutoRequestWrapper::CreateInferRequestImpl(
    const std::vector<std::shared_ptr<const ov::Node>>& inputs,
    const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    InferenceEngine::SoIInferRequestInternal request_to_share_blobs_with;
    return std::make_shared<MultiDeviceInferRequest>(inputs, outputs, request_to_share_blobs_with);
}

InferenceEngine::IInferRequestInternal::Ptr AutoRequestWrapper::CreateInferRequestImpl(
    InferenceEngine::InputsDataMap networkInputs,
    InferenceEngine::OutputsDataMap networkOutputs) {
    InferenceEngine::SoIInferRequestInternal request_to_share_blobs_with;
    return std::make_shared<MultiDeviceInferRequest>(networkInputs, networkOutputs, request_to_share_blobs_with);
}

IInferRequestInternal::Ptr AutoRequestWrapper::CreateInferRequest() {
    InferenceEngine::IExecutableNetworkInternal::Ptr exenetwork = GetExecutableNetworkInternal();
    auto syncRequestImpl = CreateInferRequestImpl(exenetwork->getInputs(), exenetwork->getOutputs());
    syncRequestImpl->setPointerToExecutableNetworkInternal(std::static_pointer_cast<MultiDeviceExecutableNetwork>(exenetwork));
    return std::make_shared<MultiDeviceAsyncInferRequest>(std::static_pointer_cast<MultiDeviceInferRequest>(syncRequestImpl),
                                                          std::static_pointer_cast<MultiDeviceExecutableNetwork>(exenetwork)->_needPerfCounters,
                                                          std::static_pointer_cast<MultiDeviceExecutableNetwork>(exenetwork),
                                                          GetCallbackExe());
}
} // namespace MultiDevicePlugin