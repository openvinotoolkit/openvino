// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime InferRequest interface
 * @file openvino/runtime/isync_infer_request.hpp
 */

#pragma once

#include <exception>
#include <memory>
#include <unordered_map>
#include <vector>

#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {

/**
 * @brief Interface for syncronous infer request
 * @ingroup ov_dev_api_sync_infer_request_api
 */
class OPENVINO_RUNTIME_API ISyncInferRequest : public IInferRequest {
public:
    /**
     * @brief Constructs syncronous inference request
     *
     * @param compiled_model pointer to compiled model
     */
    ISyncInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model);

    /**
     * @brief Gets an input/output tensor for inference.
     * @note If the tensor with the specified @p port is not found, an exception is thrown.
     * @param port Port of the tensor to get.
     * @return Tensor for the port @p port.
     */
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

    /**
     * @brief Sets an input/output tensor to infer.
     * @param port Port of the input or output tensor.
     * @param tensor Reference to a tensor. The element_type and shape of a tensor must match
     * the model's input/output element_type and size.
     */
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;

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
    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& port) const override;

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
    void set_tensors(const ov::Output<const ov::Node>& port,
                     const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

    /**
     * @brief Plugin implementation for set tensors
     *
     * @param port Port of the input tensor.
     * @param tensors Input tensors for batched infer request. The type of each tensor must match the model
     * input element type and shape (except batch dimension). Total size of tensors must match the input size.
     */
    virtual void set_tensors_impl(const ov::Output<const ov::Node> port,
                                  const std::vector<ov::SoPtr<ov::ITensor>>& tensors);

    /**
     * @brief Gets inputs for infer request
     *
     * @return vector of input ports
     */
    const std::vector<ov::Output<const ov::Node>>& get_inputs() const override;

    /**
     * @brief Gets outputs for infer request
     *
     * @return vector of output ports
     */
    const std::vector<ov::Output<const ov::Node>>& get_outputs() const override;

    /**
     * @brief Gets pointer to compiled model (usually synchronous request holds the compiled model)
     *
     * @return Pointer to the compiled model
     */
    const std::shared_ptr<const ov::ICompiledModel>& get_compiled_model() const override;

protected:
    struct FoundPort {
        size_t idx;
        enum class Type { NOT_FOUND = 0, INPUT, OUTPUT } type;

        bool found() {
            return type != Type::NOT_FOUND;
        }
        bool is_input() {
            return type == Type::INPUT;
        }
        bool is_output() {
            return !is_input();
        }
    };

    /**
     * @brief Finds input or output port
     * @return structure which contains index of Input/Output or report that port wasn't found
     */
    FoundPort find_port(const ov::Output<const ov::Node>& port) const;

    /**
     * @brief Converts batched tensors to tensor
     */
    void convert_batched_tensors();
    /**
     * @brief Basic checks for input/output tensor
     *
     * @param port Input/Output port
     * @param tensor Input/Output tensor
     */
    void check_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) const;

    /**
     * @brief Check that all tensors are valid. Throws an exception if it's not.
     */
    void check_tensors() const override;

    /**
     * @brief Allocate tensor with using custom allocator
     *
     * @param port input/output port for tensor
     * @param allocate_callback function which allocates the tensor
     */
    void allocate_tensor(const ov::Output<const ov::Node>& port,
                         const std::function<void(ov::SoPtr<ov::ITensor>& tensor)>& allocate_callback);

    std::unordered_map<std::shared_ptr<ov::descriptor::Tensor>, std::vector<ov::SoPtr<ov::ITensor>>> m_batched_tensors;
    ov::SoPtr<ov::ITensor>& get_tensor_ptr(const ov::Output<const ov::Node>& port) const;

private:
    std::shared_ptr<const ov::ICompiledModel> m_compiled_model;
    // Mutable to return reference to ov::Tensor
    mutable std::unordered_map<std::shared_ptr<descriptor::Tensor>,
                               ov::SoPtr<ov::ITensor>,
                               descriptor::TensorExtension::Hasher,
                               descriptor::TensorExtension::Equal>
        m_tensors;
    // Cache ports
    mutable std::unordered_map<size_t, FoundPort> m_cached_ports;
    mutable std::mutex m_cache_mutex;
};

};  // namespace ov
