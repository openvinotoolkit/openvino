// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides wrapper classes for infer requests and callbacks.
 *
 * @file infer_request.hpp
 */
#pragma once

#include <map>
#include <memory>
#include <string>

#include "common.hpp"
#include "profiling_info.hpp"
#include "variable_state.hpp"

namespace InferenceEngine {
class IInferRequestInternal;
class Blob;
}  // namespace InferenceEngine

namespace ov {
namespace runtime {
/**
 * @brief This is an interface of asynchronous infer request
 *
 * It can throw exceptions safely for the application, where it is properly handled.
 */
class INFERENCE_ENGINE_API_CLASS(InferRequest) {
    std::shared_ptr<SharedObject> _so;
    std::shared_ptr<ie::IInferRequestInternal> _impl;

    /**
     * @brief Constructs InferRequest from the initialized std::shared_ptr
     * @param so Plugin to use. This is required to ensure that InferRequest can work properly even if plugin object is
     * destroyed.
     * @param impl Initialized shared pointer
     */
    InferRequest(const std::shared_ptr<SharedObject>& so, const std::shared_ptr<ie::IInferRequestInternal>& impl);
    friend class ExecutableNetwork;

public:
    /**
     * @brief Default constructor
     */
    InferRequest() = default;

    /**
     * @brief Sets input/output data to infer
     *
     * @note Memory allocation does not happen
     * @param name Name of input or output blob.
     * @param data Reference to input or output blob. The type of a blob must match the network input precision and
     * size.
     */
    void set_blob(const std::string& name, const std::shared_ptr<ie::Blob>& data);

    /**
     * @brief Gets input/output data for inference
     *
     * @note Memory allocation does not happen
     * @param name A name of Blob to get
     * @return A shared pointer to a Blob with a name @p name. If a blob is not found, an exception is thrown.
     */
    std::shared_ptr<ie::Blob> get_blob(const std::string& name);

    /**
     * @brief Infers specified input(s) in synchronous mode
     *
     * @note blocks all methods of InferRequest while request is ongoing (running or waiting in queue)
     *
     */
    void infer();

    /**
     * @brief Cancels inference request
     */
    void cancel();

    /**
     * @brief Queries performance measures per layer to get feedback of what is the most time consuming layer
     *
     * @note not all plugins provide meaningful data
     * @return Vector of profiling information for layers in network
     */
    std::vector<ProfilingInfo> get_profiling_info() const;

    /**
     * @brief Sets input data to infer
     *
     * @note Memory allocation doesn't happen
     * @param inputs A reference to a map of input blobs accessed by input names.
     *        The type of Blob must correspond to the network input precision and size.
     */
    void set_input(const std::map<std::string, std::shared_ptr<ie::Blob>>& inputs);

    /**
     * @brief Sets data that will contain result of the inference
     *
     * @note Memory allocation doesn't happen
     * @param results - a reference to a map of result blobs accessed by output names.
     *        The type of Blob must correspond to the network output precision and size.
     */
    void set_output(const std::map<std::string, std::shared_ptr<ie::Blob>>& results);

    /**
     * @brief Sets new batch size when dynamic batching is enabled in executable network that created this request.
     *
     * @param batch new batch size to be used by all the following inference calls for this request.
     */
    void set_batch(const int batch);

    /**
     * @brief Start inference of specified input(s) in asynchronous mode
     *
     * @note It returns immediately. Inference starts also immediately.
     */
    void start_async();

    /**
     * @brief Waits for the result to become available. Blocks until the result
     * becomes available
     */
    void wait();

    /**
     * @brief Waits for the result to become available. Blocks until specified timeout has elapsed or the result
     * becomes available, whichever comes first.
     *
     * @param timeout Maximum duration in milliseconds to block for
     * @return true if inference request is ready and false otherwise
     */
    bool wait_for(const std::chrono::milliseconds timeout);

    /**
     * @brief Sets a callback function that will be called on success or failure of asynchronous request
     *
     * @param callback callback object which will be called on when inference finish.
     */
    void set_callback(std::function<void(std::exception_ptr)> callback);

    /**
     * @brief Gets state control interface for given infer request.
     *
     * State control essential for recurrent networks
     * @return A vector of Memory State objects
     */
    std::vector<VariableState> query_state();

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
}  // namespace runtime
}  // namespace ov
