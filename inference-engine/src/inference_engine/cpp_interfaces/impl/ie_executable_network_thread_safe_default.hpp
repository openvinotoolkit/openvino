// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cpp_interfaces/base/ie_infer_async_request_base.hpp"
#include "cpp_interfaces/ie_task_executor.hpp"
#include "cpp_interfaces/impl/ie_executable_network_internal.hpp"
#include "cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp"
#include "cpp_interfaces/impl/ie_infer_request_internal.hpp"

namespace InferenceEngine {

class ExecutableNetworkThreadSafeDefault : public ExecutableNetworkInternal,
                                           public std::enable_shared_from_this<ExecutableNetworkThreadSafeDefault> {
public:
    typedef std::shared_ptr<ExecutableNetworkThreadSafeDefault> Ptr;

    ExecutableNetworkThreadSafeDefault() {
        _taskExecutor = std::make_shared<TaskExecutor>();
        _callbackExecutor = std::make_shared<TaskExecutor>();
    }

    /**
     * @brief Create a synchronous inference request object used to infer the network
     */
    virtual InferRequestInternal::Ptr CreateInferRequestImpl(InputsDataMap networkInputs,
                                                             OutputsDataMap networkOutputs) = 0;

    /**
     * @brief Given optional implementation of creating async infer request to avoid need for it to be implemented by
     * plugin
     * @param asyncRequest - shared_ptr for the created async request
     */
    void CreateInferRequest(IInferRequest::Ptr& asyncRequest) override {
        auto syncRequestImpl = this->CreateInferRequestImpl(_networkInputs, _networkOutputs);
        syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
        auto asyncTreadSafeImpl =
            std::make_shared<AsyncInferRequestThreadSafeDefault>(syncRequestImpl, _taskExecutor, _callbackExecutor);
        asyncRequest.reset(new InferRequestBase<AsyncInferRequestThreadSafeDefault>(asyncTreadSafeImpl),
                           [](IInferRequest* p) {
                               p->Release();
                           });
        asyncTreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
    }

protected:
    ITaskExecutor::Ptr _taskExecutor = nullptr;
    ITaskExecutor::Ptr _callbackExecutor = nullptr;
};

}  // namespace InferenceEngine
