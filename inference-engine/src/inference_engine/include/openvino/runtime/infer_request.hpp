// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides wrapper classes for infer requests and callbacks.
 *
 * @file openvino/runtime/infer_request.hpp
 */
#pragma once

#include <map>
#include <memory>
#include <string>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/variable_state.hpp"

namespace InferenceEngine {
class IInferRequestInternal;
}  // namespace InferenceEngine

namespace ov {
namespace runtime {

class ExecutableNetwork;

/**
 * @brief This is an interface of asynchronous infer request
 *
 * It can throw exceptions safely for the application, where it is properly handled.
 */
class OPENVINO_RUNTIME_API InferRequest {
    std::shared_ptr<void> _so;
    std::shared_ptr<ie::IInferRequestInternal> _impl;

    /**
     * @brief Constructs InferRequest from the initialized std::shared_ptr
     * @param so Plugin to use. This is required to ensure that InferRequest can work properly even if plugin object is
     * destroyed.
     * @param impl Initialized shared pointer
     */
    InferRequest(const std::shared_ptr<void>& so, const std::shared_ptr<ie::IInferRequestInternal>& impl);
    friend class ov::runtime::ExecutableNetwork;

public:
    /**
     * @brief Default constructor
     */
    InferRequest() = default;

    /**
     * @brief Sets input/output data to infer
     *
     * @note Memory allocation does not happen
     * @param tensor Reference to input or output tensor. The type of a tensor must match the network input/output
     * precision and size.
     * @param name Name of input or output tensor.
     */
    void set_tensor1(const std::string& name, const Tensor& tensor);
    /**
     * @brief Sets input data to infer
     *
     * @note Memory allocation does not happen
     * @param idx Index of input tensor.
     * @param tensor Reference to input tensor. The type of a tensor must match the network input precision and size.
     */
    void set_input_tensor(size_t idx, const Tensor& tensor);
    /**
     * @brief Sets input data to infer
     *
     * @note Memory allocation does not happen
     * @param tensor Reference to input tensor. If model has several inputs, an exception is thrown.
     */
    void set_input_tensor(const Tensor& tensor);
    /**
     * @brief Sets output data to infer
     *
     * @note Memory allocation does not happen
     * @param idx Index of output tensor.
     * @param tensor Reference to output tensor. The type of a tensor must match the network output precision and size.
     */
    void set_output_tensor(size_t idx, const Tensor& tensor);
    /**
     * @brief Sets output data to infer
     *
     * @note Memory allocation does not happen
     * @param tensor Reference to output tensor. If model has several outputs, an exception is thrown.
     */
    void set_output_tensor(const Tensor& tensor);

    /**
     * @brief Gets input/output data for inference
     *
     * @note Memory allocation does not happen
     * @param name A name of tensor to get
     * @return A Tensor with a name @p name. If a tensor is not found, an exception is thrown.
     */
    Tensor get_tensor1(const std::string& name);
    /**
     * @brief Gets input data for inference
     *
     * @note Memory allocation does not happen
     * @param idx An index of tensor to get
     * @return A Tensor with an input index @p idx. If a tensor is not found, an exception is thrown.
     */
    Tensor get_input_tensor(size_t idx);
    /**
     * @brief Gets input data for inference
     *
     * @note Memory allocation does not happen
     * @return A Tensor with an input index @p idx. If model has several inputs, an exception is thrown.
     */
    Tensor get_input_tensor();
    /**
     * @brief Gets output data for inference
     *
     * @note Memory allocation does not happen
     * @param idx An index of tensor to get
     * @return A Tensor with an output index @p idx. If a tensor is not found, an exception is thrown.
     */
    Tensor get_output_tensor(size_t idx);
    /**
     * @brief Gets output data for inference
     *
     * @note Memory allocation does not happen
     * @return A Tensor with an output index @p idx. If model has several outputs, an exception is thrown.
     */
    Tensor get_output_tensor();

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
