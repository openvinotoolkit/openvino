// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>

#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"
#include "threading/ie_thread_safe_containers.hpp"
#include "threading/ie_itask_executor.hpp"
#include "threading/ie_executor_manager.hpp"
#include "ie_icore.hpp"
#include "request_wrapper.hpp"
#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {

class MultiDeviceInferencePlugin;
class AutoRequestWrapper : public RequestWrapper  {
public:
    using Ptr = std::shared_ptr<AutoRequestWrapper>;
    AutoRequestWrapper() = default;
    AutoRequestWrapper(InferenceEngine::SoExecutableNetworkInternal& exenetwork) : RequestWrapper(exenetwork) {}
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest(bool flag) override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap& networkInputs,
            InferenceEngine::OutputsDataMap& networkOutputs);
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
            const std::vector<std::shared_ptr<const ov::Node>>& outputs);
    ~AutoRequestWrapper() {}
};

}  // namespace MultiDevicePlugin
