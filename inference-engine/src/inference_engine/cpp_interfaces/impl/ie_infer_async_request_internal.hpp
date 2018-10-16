// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <map>
#include <string>
#include <cpp_interfaces/ie_task.hpp>
#include "cpp_interfaces/interface/ie_iinfer_async_request_internal.hpp"
#include "cpp_interfaces/impl/ie_infer_request_internal.hpp"

namespace InferenceEngine {

/**
 * @brief minimum API to be implemented by plugin, which is used in InferRequestBase forwarding mechanism
 */
class AsyncInferRequestInternal : public IAsyncInferRequestInternal, public InferRequestInternal {
public:
    typedef std::shared_ptr<AsyncInferRequestInternal> Ptr;

    explicit AsyncInferRequestInternal(InputsDataMap networkInputs, OutputsDataMap networkOutputs)
            : InferRequestInternal(networkInputs, networkOutputs), _callback(nullptr) {}

    void SetCompletionCallback(InferenceEngine::IInferRequest::CompletionCallback callback) override {
        _callback = callback;
    }

    void GetUserData(void **data) override {
        if (data == nullptr) THROW_IE_EXCEPTION << NOT_ALLOCATED_str;
        *data = _userData;
    }

    void SetUserData(void *data) override {
        _userData = data;
    }

    /**
     * @brief Set weak pointer to the corresponding public interface: IInferRequest. This allow to pass it to
     * IInferRequest::CompletionCallback
     * @param ptr - weak pointer to InferRequestBase
     */
    void SetPublicInterfacePtr(IInferRequest::Ptr ptr) {
        _publicInterface = ptr;
    }

    /**
     * @brief The minimal infer function to be implemented by plugins. It starts inference of specified input(s) in asynchronous mode
     * @note: It returns immediately. Inference starts also immediately.
     */
    virtual void StartAsyncImpl() = 0;

    void StartAsync() override {
        checkBlobs();
        StartAsyncImpl();
    };

protected:
    IInferRequest::WeakPtr _publicInterface;
    InferenceEngine::IInferRequest::CompletionCallback _callback;
    void *_userData;
};

}  // namespace InferenceEngine
