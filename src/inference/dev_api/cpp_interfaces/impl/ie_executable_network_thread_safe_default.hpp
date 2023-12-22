// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "openvino/runtime/threading/cpu_streams_executor.hpp"

namespace ov {
namespace threading {

/**
 * @brief This class provides optimal thread safe default implementation.
 * The class is recommended to be used as a base class for Executable Network implementation during plugin development.
 * @ingroup ie_dev_api_exec_network_api
 */
class INFERENCE_ENGINE_1_0_DEPRECATED ExecutableNetworkThreadSafeDefault : public IExecutableNetworkInternal {
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
    explicit ExecutableNetworkThreadSafeDefault(
        const ITaskExecutor::Ptr& taskExecutor = std::make_shared<CPUStreamsExecutor>(IStreamsExecutor::Config{
            "Default"}),
        const ITaskExecutor::Ptr& callbackExecutor = std::make_shared<CPUStreamsExecutor>(IStreamsExecutor::Config{
            "Callback"}))
        : _taskExecutor{taskExecutor},
          _callbackExecutor{callbackExecutor} {}

    /**
     * @brief Given optional implementation of creating asynchronous inference request to avoid
     * need for it to be implemented by plugin
     * @return shared_ptr for the created asynchronous inference request
     */
    IInferRequestInternal::Ptr CreateInferRequest() override {
        return CreateAsyncInferRequestFromSync();
    }

protected:
    /**
     * @brief Creates asyncronous inference request from synchronous request returned by CreateInferRequestImpl
     * @tparam AsyncInferRequestType A type of asynchronous inference request to use a wrapper for synchronous request
     * @return A shared pointer to an asynchronous inference request
     */
    template <typename AsyncInferRequestType = AsyncInferRequestThreadSafeDefault>
    IInferRequestInternal::Ptr CreateAsyncInferRequestFromSync() {
        InferenceEngine::IInferRequestInternal::Ptr syncRequestImpl;
        try {
            syncRequestImpl = this->CreateInferRequestImpl(_parameters, _results);
        } catch (const ::ov::NotImplemented&) {
        } catch (const InferenceEngine::NotImplemented&) {
        }
        if (!syncRequestImpl) {
            syncRequestImpl = this->CreateInferRequestImpl(_networkInputs, _networkOutputs);
            syncRequestImpl->setModelInputsOutputs(_parameters, _results);
        }
        syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
        return std::make_shared<AsyncInferRequestType>(syncRequestImpl, _taskExecutor, _callbackExecutor);
    }

    ITaskExecutor::Ptr _taskExecutor = nullptr;      //!< Holds a task executor
    ITaskExecutor::Ptr _callbackExecutor = nullptr;  //!< Holds a callback executor
};

}  // namespace threading
}  // namespace ov
