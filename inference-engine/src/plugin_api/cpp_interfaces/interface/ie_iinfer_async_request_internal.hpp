// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iinfer_request.hpp>
#include <map>
#include <memory>
#include <string>

#include "ie_iinfer_request_internal.hpp"

namespace InferenceEngine {

/**
 * @interface IAsyncInferRequestInternal
 * @ingroup ie_dev_api_async_infer_request_api
 * @brief An internal API of asynchronous inference request to be implemented by plugin,
 * which is used in InferRequestBase forwarding mechanism
 */
class IAsyncInferRequestInternal : virtual public IInferRequestInternal {
public:
    /**
     * @brief A shared pointer to IAsyncInferRequestInternal interface
     */
    typedef std::shared_ptr<IAsyncInferRequestInternal> Ptr;

    /**
     * @brief A virtual destructor
     */
    virtual ~IAsyncInferRequestInternal() = default;

    /**
     * @brief Start inference of specified input(s) in asynchronous mode
     * @note The method returns immediately. Inference starts also immediately.
     */
    virtual void StartAsync() = 0;

    /**
     * @brief Waits for the result to become available. Blocks until specified millis_timeout has elapsed or the result
     * becomes available, whichever comes first.
     * @param millis_timeout - maximum duration in milliseconds to block for
     * @note There are special cases when millis_timeout is equal some value of WaitMode enum:
     * * STATUS_ONLY - immediately returns request status (IInferRequest::RequestStatus). It doesn't block or interrupt
     * current thread.
     * * RESULT_READY - waits until inference result becomes available
     * @return A status code
     */
    virtual StatusCode Wait(int64_t millis_timeout) = 0;

    /**
     * @brief Get arbitrary data for the request
     * @param data A pointer to a pointer to arbitrary data
     */
    virtual void GetUserData(void** data) = 0;

    /**
     * @brief Set arbitrary data for the request
     * @param data A pointer to a pointer to arbitrary data
     */
    virtual void SetUserData(void* data) = 0;

    /**
     * @brief Set callback function which will be called on success or failure of asynchronous request
     * @param callback - function to be called with the following description:
     */
    virtual void SetCompletionCallback(IInferRequest::CompletionCallback callback) = 0;
};

}  // namespace InferenceEngine
