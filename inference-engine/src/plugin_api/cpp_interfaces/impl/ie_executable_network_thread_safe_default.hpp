// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cpp_interfaces/base/ie_infer_async_request_base.hpp"
#include "cpp_interfaces/impl/ie_executable_network_internal.hpp"
#include "cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp"
#include "cpp_interfaces/impl/ie_infer_request_internal.hpp"
#include "threading/ie_cpu_streams_executor.hpp"

namespace InferenceEngine {

/**
 * @brief This class provides optimal thread safe default implementation.
 * The class is recommended to be used as a base class for Executable Network impleentation during plugin development.
 * @ingroup ie_dev_api_exec_network_api
 */
class ExecutableNetworkThreadSafeDefault : public ExecutableNetworkInternal,
                                           public std::enable_shared_from_this<ExecutableNetworkThreadSafeDefault> {
public:
    /**
     * @brief A shared pointer to a ExecutableNetworkThreadSafeDefault object
     */
    typedef std::shared_ptr<ExecutableNetworkThreadSafeDefault> Ptr;

    /**
     * @brief      Constructs a new instance.
     *
     * @param[in]  taskExecutor      The task executor used
     * @param[in]  callbackExecutor  The callback executor
     */
    explicit
    ExecutableNetworkThreadSafeDefault(const ITaskExecutor::Ptr& taskExecutor
                                             = std::make_shared<CPUStreamsExecutor>(IStreamsExecutor::Config{"Default"}),
                                       const ITaskExecutor::Ptr& callbackExecutor
                                             = std::make_shared<CPUStreamsExecutor>(IStreamsExecutor::Config{"Callback"})) :
        _taskExecutor{taskExecutor},
        _callbackExecutor{callbackExecutor} {
    }

    /**
     * @brief Given optional implementation of creating asynchnous inference request to avoid
     * need for it to be implemented by plugin
     * @param asyncRequest shared_ptr for the created asynchnous inference request
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

    /**
     * @brief Gets the executor.
     * @return The executor.
     */
    ITaskExecutor::Ptr& GetExecutor() {
        return _taskExecutor;
    }

protected:
    /**
     * @brief Create a synchronous inference request object used to infer the network
     * @note Used by ExecutableNetworkThreadSafeDefault::CreateInferRequest as a plugin-specific implementation
     * @param networkInputs An input info map needed to create input blobs
     * @param networkOutputs An output data map needed to create output blobs
     * @return Synchronous inference request object
     */
    virtual InferRequestInternal::Ptr CreateInferRequestImpl(InputsDataMap networkInputs,
                                                             OutputsDataMap networkOutputs) = 0;

    ITaskExecutor::Ptr _taskExecutor = nullptr;  //!< Holds a task executor
    ITaskExecutor::Ptr _callbackExecutor = nullptr;  //!< Holds a callback executor
};

}  // namespace InferenceEngine
