// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides InferRequest.
 *
 * @file openvino/runtime/infer_request.hpp
 */
#pragma once

#include <map>
#include <memory>
#include <string>

#include "openvino/core/node_output.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/variable_state.hpp"

namespace InferenceEngine {
class IInferRequestInternal;
}  // namespace InferenceEngine

namespace ov {
namespace runtime {

class CompiledModel;

/**
 * @brief This is a class of infer request which can be run in asynchronous or synchronous manners.
 */
class OPENVINO_RUNTIME_API InferRequest {
    std::shared_ptr<InferenceEngine::IInferRequestInternal> _impl;
    std::shared_ptr<void> _so;

    /**
     * @brief Constructs InferRequest from the initialized std::shared_ptr
     * @param impl Initialized shared pointer
     * @param so Plugin to use. This is required to ensure that InferRequest can work properly even if plugin object is
     * destroyed.
     */
    InferRequest(const std::shared_ptr<InferenceEngine::IInferRequestInternal>& impl, const std::shared_ptr<void>& so);
    friend class ov::runtime::CompiledModel;

public:
    /**
     * @brief Default constructor
     */
    InferRequest() = default;

    /**
     * @brief Default copy constructor
     * @param other other InferRequest object
     */
    InferRequest(const InferRequest& other) = default;

    /**
     * @brief Default copy assignment operator
     * @param other Another InferRequest object
     * @return A reference to the current object
     */
    InferRequest& operator=(const InferRequest& other) = default;

    /**
     * @brief Default move constructor
     * @param other other InferRequest object
     */
    InferRequest(InferRequest&& other) = default;

    /**
     * @brief Default move assignment operator
     * @param other other InferRequest object
     * @return reference to the current object
     */
    InferRequest& operator=(InferRequest&& other) = default;

    /**
     * @brief Destructor preserves unloading order of implementation object and reference to library
     * @note To preserve destruction order inside default generated assignment operator we store `_impl` before `_so`.
     *       And use destructor to remove implementation object before reference to library explicitly
     */
    ~InferRequest();

    /**
     * @brief Sets input/output tensor to infer on
     *
     * @param tensor_name Name of input or output tensor.
     * @param tensor Reference to a tensor. The element_type and shape of a tensor must match
     * the model's input/output element_type and size.
     */
    void set_tensor(const std::string& tensor_name, const Tensor& tensor);

    /**
     * @brief Sets input/output tensor to infer
     * @param port Port of input or output tensor. Note, that the ports get from the following methods can be used:
     * - ov::Model::input()
     * - ov::Model::inputs()
     * - ov::Model::outputs()
     * - ov::Model::outputs()
     * - ov::runtime::CompiledModel::input()
     * - ov::runtime::CompiledModel::inputs()
     * - ov::runtime::CompiledModel::outputs()
     * - ov::runtime::CompiledModel::outputs()
     * @param tensor Reference to a tensor. The element_type and shape of a tensor must match
     * the model's input/output element_type and size.
     */
    void set_tensor(const ov::Output<const ov::Node>& port, const Tensor& tensor);

    /**
     * @brief Sets input/output tensor to infer
     * @param port Port of input or output tensor. Note, that the ports get from the following methods can be used:
     * - ov::Model::input()
     * - ov::Model::inputs()
     * - ov::Model::outputs()
     * - ov::Model::outputs()
     * - ov::runtime::CompiledModel::input()
     * - ov::runtime::CompiledModel::inputs()
     * - ov::runtime::CompiledModel::outputs()
     * - ov::runtime::CompiledModel::outputs()
     * @param tensor Reference to a tensor. The element_type and shape of a tensor must match
     * the model's input/output element_type and size.
     */
    void set_tensor(const ov::Output<ov::Node>& port, const Tensor& tensor);

    /**
     * @brief Sets batch of tensors for input data to infer by tensor name
     * Model input shall have batch dimension and number of @p tensors shall match with batch size
     * Current version supports set tensors to model inputs only. In case if @p tensor_name is associated
     * with output (or any other non-input node) - an exception will be thrown
     *
     * @param tensor_name Name of input tensor.
     * @param tensors Input tensors for batched infer request. The type of each tensor must match the model
     * input element type and shape (except batch dimension). Total size of tensors shall match with input's size
     */
    void set_tensors(const std::string& tensor_name, const std::vector<Tensor>& tensors);

    /**
     * @brief Sets batch of tensors for input data to infer by input port
     * Model input shall have batch dimension and number of @p tensors shall match with batch size
     * Current version supports set tensors to model inputs only. In case if @p port is associated
     * with output (or any other non-input node) - an exception will be thrown
     *
     * @param port Port of input tensor.
     * @param tensors Input tensors for batched infer request. The type of each tensor must match the model
     * input element type and shape (except batch dimension). Total size of tensors shall match with input's size
     */
    void set_tensors(const ov::Output<const ov::Node>& port, const std::vector<Tensor>& tensors);

    /**
     * @brief Sets input tensor to infer
     *
     * @param idx Index of input tensor. If @p idx is greater than number of model inputs, an exception is thrown
     * @param tensor Reference to a tensor. The element_type and shape of a tensor must match
     * the model's input/output element_type and size.
     */
    void set_input_tensor(size_t idx, const Tensor& tensor);

    /**
     * @brief Sets input tensor to infer models with single input
     * @note If model has several inputs, an exception is thrown
     * @param tensor Reference to input tensor.
     */
    void set_input_tensor(const Tensor& tensor);

    /**
     * @brief Sets batch of tensors for single input data
     * Model input shall have batch dimension and number of @p tensors shall match with batch size
     *
     * @param tensors Input tensors for batched infer request. The type of each tensor must match the model
     * input element type and shape (except batch dimension). Total size of tensors shall match with input's size
     */
    void set_input_tensors(const std::vector<Tensor>& tensors);

    /**
     * @brief Sets batch of tensors for input data to infer by input name
     * Model input shall have batch dimension and number of @p tensors shall match with batch size
     *
     * @param idx Name of input tensor.
     * @param tensors Input tensors for batched infer request. The type of each tensor must match the model
     * input element type and shape (except batch dimension). Total size of tensors shall match with input's size
     */
    void set_input_tensors(size_t idx, const std::vector<Tensor>& tensors);

    /**
     * @brief Sets output tensor to infer
     * @note An index of input preserved accross ov::Model, ov::runtime::CompiledModel and ov::runtime::InferRequest
     * @param idx Index of output tensor.
     * @param tensor Reference to output tensor. The type of a tensor must match the model output element type and
     * shape.
     */
    void set_output_tensor(size_t idx, const Tensor& tensor);

    /**
     * @brief Sets output tensor to infer models with single output
     * @note If model has several outputs, an exception is thrown.
     * @param tensor Reference to output tensor.
     */
    void set_output_tensor(const Tensor& tensor);

    /**
     * @brief Gets input/output tensor for inference by tensor name
     * @param tensor_name A name of tensor to get
     * @return A Tensor with a name @p tensor_name. If a tensor is not found, an exception is thrown.
     */
    Tensor get_tensor(const std::string& tensor_name);

    /**
     * @brief Gets input/output tensor for inference
     * @note If a tensor with specified @p port is not found, an exception is thrown
     * @param port Port of tensor to get
     * @return A Tensor for the port @p port.
     */
    Tensor get_tensor(const ov::Output<const ov::Node>& port);

    /**
     * @brief Gets input/output tensor for inference
     * @note If a tensor with specified @p port is not found, an exception is thrown
     * @param port Port of tensor to get
     * @return A Tensor for the port @p port.
     */
    Tensor get_tensor(const ov::Output<ov::Node>& port);

    /**
     * @brief Gets input tensor for inference
     *
     * @param idx An index of tensor to get
     * @return A Tensor with an input index @p idx. If a tensor with specified @p idx is not found, an exception is
     * thrown.
     */
    Tensor get_input_tensor(size_t idx);

    /**
     * @brief Gets input tensor for inference
     *
     * @return An input Tensor for the model. If model has several inputs, an exception is thrown.
     */
    Tensor get_input_tensor();

    /**
     * @brief Gets output tensor for inference
     *
     * @param idx An index of tensor to get
     * @return A Tensor with an output index @p idx. If a tensor with specified @p idx is not found, an exception is
     * thrown.
     */
    Tensor get_output_tensor(size_t idx);

    /**
     * @brief Gets output tensor for inference
     *
     * @return An output Tensor for the model. If model has several outputs, an exception is thrown.
     */
    Tensor get_output_tensor();

    /**
     * @brief Infers specified input(s) in synchronous mode
     * @note blocks all methods of InferRequest while request is ongoing (running or waiting in queue)
     *       Calling any method will lead to throwning ov::runtime::Busy exception
     */
    void infer();

    /**
     * @brief Cancels inference request
     */
    void cancel();

    /**
     * @brief Queries performance measures per layer to get feedback of what is the most time consuming operation
     * @note not all plugins provide meaningful data
     * @return Vector of profiling information for operations in model
     */
    std::vector<ProfilingInfo> get_profiling_info() const;

    /**
     * @brief Starts inference of specified input(s) in asynchronous mode
     * @note It returns immediately. Inference starts also immediately.
     *       Calling any method while the request in a running state will lead to throwning ov::runtime::Busy exception
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
     * @brief Sets a callback std::function that will be called on success or failure of asynchronous request
     * @param callback callback object which will be called on when inference finish.
     */
    void set_callback(std::function<void(std::exception_ptr)> callback);

    /**
     * @brief Gets state control interface for given infer request.
     *
     * State control essential for recurrent models
     * @return A vector of Variable State objects
     */
    std::vector<VariableState> query_state();

    /**
     * @brief Returns compiled model that creates this inference request
     * @return Compiled model object
     */
    CompiledModel get_compiled_model();

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
     * @param other Another inference request
     * @return true if current InferRequest object doesn't wrap the same impl as the operator's arg
     */
    bool operator!=(const InferRequest& other) const noexcept;

    /**
     * @brief Compares whether this request wraps the same impl underneath
     * @param other Another inference request
     * @return true if current InferRequest object wraps the same impl as the operator's arg
     */
    bool operator==(const InferRequest& other) const noexcept;
};
}  // namespace runtime
}  // namespace ov
