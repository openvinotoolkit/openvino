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

#include "cpp/ie_memory_state.hpp"
#include "ie_remote_context.hpp"
#include "ie_iinfer_request.hpp"
#include "details/ie_so_loader.h"
#include "ie_blob.h"

namespace InferenceEngine {
namespace details {
class SharedObjectLoader;
}
class IInferRequest;
class IInferRequestInternal;

/**
 * @copybrief IInferRequest
 *
 * Wraps IInferRequest
 * It can throw exceptions safely for the application, where it is properly handled.
 */
class InferRequest {
    std::shared_ptr<IInferRequestInternal>          _impl;
    details::SharedObjectLoader                     _so;

    explicit InferRequest(const std::shared_ptr<IInferRequestInternal>&         impl,
                          const std::shared_ptr<details::SharedObjectLoader>&   so);

    friend class ExecutableNetwork;

public:
    /**
     * @brief A smart pointer to the InferRequest object
     */
    using Ptr = std::shared_ptr<InferRequest>;

    /**
     * @brief Default constructor
     */
    InferRequest() = default;

    /**
     * @brief Destructor
     */
    ~InferRequest() = default;

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
     * @copybrief IInferRequest::GetBlob
     *
     * Wraps IInferRequest::GetBlob
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
     * @copybrief IInferRequest::Infer
     * @note blocks all methods of InferRequest while request is ongoing (running or waiting in queue)
     *
     * Wraps IInferRequest::Infer
     */
    void Infer();

    /**
     * @brief Cancel inference request
     * @param name Name of input blob.
     * @return pointer to pre-process info of blob with name
     */
    void Cancel();

    /**
     * @copybrief IInferRequest::GetPerformanceCounts
     *
     * Wraps IInferRequest::GetPerformanceCounts
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
     * @copybrief IInferRequest::Wait
     *
     * Wraps IInferRequest::Wait
     * @param millis_timeout Maximum duration in milliseconds to block for
     * @note There are special cases when millis_timeout is equal some value of the WaitMode enum:
     * * STATUS_ONLY - immediately returns inference status (IInferRequest::RequestStatus). It does not block or
     * interrupt current thread
     * * RESULT_READY - waits until inference result becomes available
     * @return A status code of operation
     */
    StatusCode Wait(int64_t millis_timeout);

    /**
     * @copybrief IInferRequest::SetCompletionCallback
     *
     * Wraps IInferRequest::SetCompletionCallback
     *
     * @param callbackToSet Lambda callback object which will be called on processing finish.
     */
    void SetCompletionCallback(std::function<void()>);
    void SetCompletionCallback(std::function<void(InferRequest, StatusCode)>);
    using CompletionCallback = void(std::shared_ptr<IInferRequest>, StatusCode);
    INFERENCE_ENGINE_DEPRECATED("Will be removed")
    void SetCompletionCallback(CompletionCallback*);

    /**
     * @copybrief IExecutableNetwork::QueryState
     *
     * Wraps IExecutableNetwork::QueryState
     * @return A vector of Memory State objects
     */
    std::vector<VariableState> QueryState();

    /**
     * @brief  IInferRequest pointer to be used directly in CreateInferRequest functions
     * @return A shared pointer to IInferRequest interface
     */
    INFERENCE_ENGINE_DEPRECATED("Will be removed")
    operator std::shared_ptr<IInferRequest> ();

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
};
}  // namespace InferenceEngine
