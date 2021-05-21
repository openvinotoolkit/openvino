// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides wrapper classes for infer requests and callbacks.
 *
 * @file ie_infer_request.hpp
 */
#pragma once

#include <map>
#include <memory>
#include <string>

#include "ie_blob.h"
#include "cpp/ie_memory_state.hpp"
#include "ie_iinfer_request.hpp"
#include "details/ie_so_loader.h"

namespace InferenceEngine {

class IInferRequestInternal;

namespace details {
class ICompletionCallbackWrapper;
}  // namespace details

/**
 * @copybrief IInferRequest
 *
 * Wraps IInferRequest
 * It can throw exceptions safely for the application, where it is properly handled.
 */
class INFERENCE_ENGINE_API_CLASS(InferRequest) {
    details::SharedObjectLoader                          _so;
    std::shared_ptr<IInferRequestInternal>               _impl;
    IE_SUPPRESS_DEPRECATED_START
    IInferRequest::Ptr                                   actual;
    std::shared_ptr<details::ICompletionCallbackWrapper> callback;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Constructs InferRequest from the initialized std::shared_ptr
     * @param so Plugin to use. This is required to ensure that InferRequest can work properly even if plugin object is destroyed.
     * @param impl Initialized shared pointer
     */
    InferRequest(const details::SharedObjectLoader&             so,
                 const std::shared_ptr<IInferRequestInternal>&  impl);
    friend class ExecutableNetwork;

public:
    /**
     * @enum WaitMode
     * @brief Enumeration to hold wait mode for IInferRequest
     */
    enum WaitMode : int64_t {
        /** Wait until inference result becomes available */
        RESULT_READY = -1,
        /** IInferRequest doesn't block or interrupt current thread and immediately returns inference status */
        STATUS_ONLY = 0,
    };

    /**
     * @brief A smart pointer to the InferRequest object
     */
    using Ptr = std::shared_ptr<InferRequest>;

    /**
     * @brief Default constructor
     */
    InferRequest() = default;

    IE_SUPPRESS_DEPRECATED_START
    /**
     * @deprecated This ctor will be removed in 2022.1
     * @brief Constructs InferRequest from the initialized std::shared_ptr
     * @param request Initialized shared pointer
     * @param splg Plugin to use. This is required to ensure that InferRequest can work properly even if plugin object is destroyed.
     */
    INFERENCE_ENGINE_DEPRECATED("This ctor will be removed in 2022.1")
    explicit InferRequest(IInferRequest::Ptr request,
                          std::shared_ptr<details::SharedObjectLoader> splg = {});
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Sets input/output data to infer
     *
     * @note Memory allocation does not happen
     * @param name Name of input or output blob.
     * @param data Reference to input or output blob. The type of a blob must match the network input precision and
     * size.
     */
    void SetBlob(const std::string& name, const Blob::Ptr& data);

    /**
     * @brief Gets input/output data for inference
     *
     * @note Memory allocation does not happen
     * @param name A name of Blob to get
     * @return A shared pointer to a Blob with a name @p name. If a blob is not found, an exception is thrown.
     */
    Blob::Ptr GetBlob(const std::string& name);

    /**
     * @brief Sets blob with a pre-process information
     * @note Returns an error in case if data blob is output
     * @param name Name of input blob.
     * @param data A reference to input. The type of Blob must correspond to the network input precision and size.
     * @param info Preprocess info for blob.
     */
    void SetBlob(const std::string &name, const Blob::Ptr &data, const PreProcessInfo& info);

    /**
     * @brief Gets pre-process for input data
     * @param name Name of input blob.
     * @return pointer to pre-process info of blob with name
     */
    const PreProcessInfo& GetPreProcess(const std::string& name) const;

    /**
     * @brief Infers specified input(s) in synchronous mode
     *
     * @note blocks all methods of InferRequest while request is ongoing (running or waiting in queue)
     *
     */
    void Infer();

    /**
     * @brief Cancels inference request
     */
    void Cancel();

    /**
     * @brief Queries performance measures per layer to get feedback of what is the most time consuming layer
     *
     * @note not all plugins provide meaningful data
     * @return Map of layer names to profiling information for that layer
     */
    std::map<std::string, InferenceEngineProfileInfo> GetPerformanceCounts() const;

    /**
     * @brief Sets input data to infer
     *
     * @note Memory allocation doesn't happen
     * @param inputs A reference to a map of input blobs accessed by input names.
     *        The type of Blob must correspond to the network input precision and size.
     */
    void SetInput(const BlobMap& inputs);

    /**
     * @brief Sets data that will contain result of the inference
     *
     * @note Memory allocation doesn't happen
     * @param results - a reference to a map of result blobs accessed by output names.
     *        The type of Blob must correspond to the network output precision and size.
     */
    void SetOutput(const BlobMap& results);

    /**
     * @brief Sets new batch size when dynamic batching is enabled in executable network that created this request.
     *
     * @param batch new batch size to be used by all the following inference calls for this request.
     */
    void SetBatch(const int batch);

    /**
     * @brief Start inference of specified input(s) in asynchronous mode
     *
     * @note It returns immediately. Inference starts also immediately.
     */
    void StartAsync();

    /**
     * @brief Waits for the result to become available. Blocks until specified millis_timeout has elapsed or the result
     * becomes available, whichever comes first.
     *
     *
     * @param millis_timeout Maximum duration in milliseconds to block for
     * @note There are special cases when millis_timeout is equal some value of the WaitMode enum:
     * * STATUS_ONLY - immediately returns inference status (IInferRequest::RequestStatus). It does not block or
     * interrupt current thread
     * * RESULT_READY - waits until inference result becomes available
     * @return A status code of operation
     */
    StatusCode Wait(int64_t millis_timeout = RESULT_READY);

private:
    void SetCompletionCallbackImpl(std::function<void()>);
    void SetCompletionCallbackImpl(std::function<void(InferRequest, StatusCode)>);
    IE_SUPPRESS_DEPRECATED_START
    void SetCompletionCallbackImpl(IInferRequest::CompletionCallback);
    IE_SUPPRESS_DEPRECATED_END

    template<typename T>
    struct SetCallback {
        void operator()(std::function<void()> f) {_this.SetCompletionCallbackImpl(std::move(f));}
        InferRequest& _this;
    };

public:
    /**
     * @brief Sets a callback function that will be called on success or failure of asynchronous request
     *
     * @param callbackToSet callback object which will be called on when inference finish.
     */
    template<typename F>
    void SetCompletionCallback(F callbackToSet) {
        SetCallback<F>{*this}(std::move(callbackToSet));
    }

    /**
     * @brief Gets state control interface for given infer request.
     *
     * State control essential for recurrent networks
     * @return A vector of Memory State objects
     */
    std::vector<VariableState> QueryState();

    IE_SUPPRESS_DEPRECATED_START
    /**
     * @brief  IInferRequest pointer to be used directly in CreateInferRequest functions
     * @return A shared pointer to IInferRequest interface
     */
    INFERENCE_ENGINE_DEPRECATED("Will be removed")
    operator std::shared_ptr<IInferRequest> ();
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Checks if current InferRequest object is not initialized
     * @return true if current InferRequest object is not initialized, false - otherwise
     */
    bool operator!() const noexcept;

    /**
     * @brief Checks if current InferRequest object is initialized
     * @return true if current InferRequest object is initialized, false - otherwise
     */
    explicit operator bool() const noexcept;

    /**
     * @brief Compares whether this request wraps the same impl underneath
     * @return true if current InferRequest object doesn't wrap the same impl as the operator's arg
     */
    bool operator!=(const InferRequest&) const noexcept;

    /**
     * @brief Compares whether this request wraps the same impl underneath
     * @return true if current InferRequest object wraps the same impl as the operator's arg
     */
    bool operator==(const InferRequest&) const noexcept;
};

template<>
struct InferRequest::SetCallback<std::function<void(InferRequest, StatusCode)>> {
    void operator()(std::function<void(InferRequest, StatusCode)> f) {
        _this.SetCompletionCallbackImpl(std::move(f));
    }
    InferRequest& _this;
};

IE_SUPPRESS_DEPRECATED_START

template<>
struct InferRequest::SetCallback<IInferRequest::CompletionCallback> {
    void operator()(IInferRequest::CompletionCallback f) {
        _this.SetCompletionCallbackImpl(std::move(f));
    }
    InferRequest& _this;
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace InferenceEngine
