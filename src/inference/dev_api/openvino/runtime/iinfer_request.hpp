// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime InferRequest interface
 * @file openvino/runtime/iinfer_request.hpp
 */

#pragma once

#include <exception>
#include <memory>
#include <unordered_map>
#include <vector>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {

class IAsyncInferRequest;
class ICompiledModel;

/**
 * @brief An internal API of inference request to be implemented by plugin
 * @ingroup ov_dev_api_infer_request_api
 */
class OPENVINO_RUNTIME_API IInferRequest {
public:
    virtual ~IInferRequest();

    /**
     * @brief Infers specified input(s) in synchronous mode
     * @note blocks all method of InferRequest while request is ongoing (running or waiting in queue)
     */
    virtual void infer() = 0;

    /**
     * @brief Queries performance measures per layer to identify the most time consuming operation.
     * @note Not all plugins provide meaningful data.
     * @return Vector of profiling information for operations in a model.
     */
    virtual std::vector<ov::ProfilingInfo> get_profiling_info() const = 0;

    /**
     * @brief Gets an input/output tensor for inference.
     * @note If the tensor with the specified @p port is not found, an exception is thrown.
     * @param port Port of the tensor to get.
     * @return Tensor for the port @p port.
     */
    virtual ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const = 0;

    /**
     * @brief Sets an input/output tensor to infer.
     * @param port Port of the input or output tensor.
     * @param tensor Reference to a tensor. The element_type and shape of a tensor must match
     * the model's input/output element_type and size.
     */
    virtual void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) = 0;

    /**
     * @brief Gets a batch of tensors for input data to infer by input port.
     * Model input must have batch dimension, and the number of @p tensors must match the batch size.
     * The current version supports setting tensors to model inputs only. If @p port is associated
     * with output (or any other non-input node), an exception is thrown.
     *
     * @param port Port of the input tensor.
     * @param tensors Input tensors for batched infer request. The type of each tensor must match the model
     * input element type and shape (except batch dimension). Total size of tensors must match the input size.
     * @return vector of tensors
     */
    virtual std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& port) const = 0;

    /**
     * @brief Sets a batch of tensors for input data to infer by input port.
     * Model input must have batch dimension, and the number of @p tensors must match the batch size.
     * The current version supports setting tensors to model inputs only. If @p port is associated
     * with output (or any other non-input node), an exception is thrown.
     *
     * @param port Port of the input tensor.
     * @param tensors Input tensors for batched infer request. The type of each tensor must match the model
     * input element type and shape (except batch dimension). Total size of tensors must match the input size.
     */
    virtual void set_tensors(const ov::Output<const ov::Node>& port,
                             const std::vector<ov::SoPtr<ov::ITensor>>& tensors) = 0;

    /**
     * @brief Gets state control interface for the given infer request.
     *
     * State control essential for recurrent models.
     * @return Vector of Variable State objects.
     */
    virtual std::vector<ov::SoPtr<ov::IVariableState>> query_state() const = 0;

    /**
     * @brief Gets pointer to compiled model (usually synchronous request holds the compiled model)
     *
     * @return Pointer to the compiled model
     */
    virtual const std::shared_ptr<const ov::ICompiledModel>& get_compiled_model() const = 0;

    /**
     * @brief Gets inputs for infer request
     *
     * @return vector of input ports
     */
    virtual const std::vector<ov::Output<const ov::Node>>& get_inputs() const = 0;

    /**
     * @brief Gets outputs for infer request
     *
     * @return vector of output ports
     */
    virtual const std::vector<ov::Output<const ov::Node>>& get_outputs() const = 0;

protected:
    /**
     * @brief Check that all tensors are valid. Throws an exception if it's not.
     */
    virtual void check_tensors() const = 0;
    friend IAsyncInferRequest;
};

};  // namespace ov
