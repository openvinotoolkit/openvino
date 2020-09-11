// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "cpp_interfaces/impl/ie_infer_request_internal.hpp"
#include "cpp_interfaces/interface/ie_iinfer_async_request_internal.hpp"

namespace InferenceEngine {

/**
 * @brief minimum API to be implemented by plugin, which is used in InferRequestBase forwarding mechanism
 * @ingroup ie_dev_api_async_infer_request_api
 */
class AsyncInferRequestInternal : public IAsyncInferRequestInternal, public InferRequestInternal {
public:
    /**
     * @brief A shared pointer to a AsyncInferRequestInternal implementation
     */
    typedef std::shared_ptr<AsyncInferRequestInternal> Ptr;

    /**
     * @brief      Constructs a new instance.
     * @param[in]  networkInputs   The network inputs info
     * @param[in]  networkOutputs  The network outputs data
     */
    AsyncInferRequestInternal(const InputsDataMap& networkInputs, const OutputsDataMap& networkOutputs)
        : InferRequestInternal(networkInputs, networkOutputs), _callback(nullptr), _userData(nullptr) {}

    void SetCompletionCallback(IInferRequest::CompletionCallback callback) override {
        _callback = callback;
    }

    void GetUserData(void** data) override {
        if (data == nullptr) THROW_IE_EXCEPTION << NOT_ALLOCATED_str;
        *data = _userData;
    }

    void SetUserData(void* data) override {
        _userData = data;
    }

    /**
     * @brief Set weak pointer to the corresponding public interface: IInferRequest. This allow to pass it to
     * IInferRequest::CompletionCallback
     * @param ptr A weak pointer to InferRequestBase
     */
    void SetPublicInterfacePtr(IInferRequest::Ptr ptr) {
        _publicInterface = ptr;
    }

    void StartAsync() override {
        checkBlobs();
        StartAsyncImpl();
    };

protected:
    /**
     * @brief The minimal asynchronous inference function to be implemented by plugins.
     * It starts inference of specified input(s) in asynchronous mode
     * @note
     *  * The methos is used in AsyncInferRequestInternal::StartAsync which performs common steps first and
     *  calls plugin dependent implementation of this method after.
     *  * It returns immediately. Inference starts also immediately.
     */
    virtual void StartAsyncImpl() = 0;

    IInferRequest::WeakPtr _publicInterface;  //!< A weak pointer to a IInferRequest interface for callback calling
    InferenceEngine::IInferRequest::CompletionCallback _callback;  //!< A callback
    void* _userData;  //!< A callback user data
};

}  // namespace InferenceEngine
