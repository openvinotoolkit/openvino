// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/icompiled_model.hpp"
#include "intel_npu/common/igraph.hpp"
#include "intel_npu/common/variable_state.hpp"
#include "intel_npu/network_metadata.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/iplugin.hpp"

namespace intel_npu {

/**
 * @brief Acts as an interface for the inference request structures implemented by all backends.
 * @details The operations common for all backends can be found implemented here, these include tensors' extraction,
 * state variables handling and and a few helper functions.
 * @note Regarding the code design, the "ov::ISyncInferRequest" would normally be the better option for
 * inheritance. However, the interface exposed by that class forces some additional latency in several unfavorable
 * scenarios, thus a reimplementation was required.
 */
class SyncInferRequest : public ov::IInferRequest {
public:
    explicit SyncInferRequest(const std::shared_ptr<const ICompiledModel>& compiledModel, const Config& config);

    /**
     * @brief Gets an input/output tensor for inference.
     * @note If the tensor with the specified @p port is not found, an exception is thrown.
     * @param port Port of the tensor to get.
     * @return Tensor for the port @p port.
     */
    virtual ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

    /**
     * @brief Sets an input/output tensor to infer.
     * @param port Port of the input or output tensor.
     * @param tensor Reference to a tensor. The element_type and shape of a tensor must match
     * the model's input/output element_type and size.
     */
    virtual void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;

    /**
     * @brief Currently there is no support implemented for batches of tensors, thus this call is a simple redirection
     * to the "get_tensor" one.
     */
    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& port) const override;

    /**
     * @brief Currently there is no support implemented for batches of tensors, thus this call is a simple redirection
     * to the "set_tensor" one.
     */
    virtual void set_tensors(const ov::Output<const ov::Node>& port,
                             const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

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

    /**
     * @brief Used for executing the inference.
     */
    virtual void infer_async() = 0;

    /**
     * @brief Used for retrieving the prediction's result.
     */
    virtual void get_result() = 0;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

    /**
     * @brief Initializes the tensor values corresponding to the state variables.
     * @details The inital values are usually all 0s.
     */
    void initialize_states();

protected:
    /**
     * @see ov::ISyncInferRequest
     */
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
     * @see ov::ISyncInferRequest
     */
    FoundPort find_port(const ov::Output<const ov::Node>& port) const;

    /**
     * @brief Basic checks for input/output tensor
     *
     * @param port Input/Output port
     * @param tensor Input/Output tensor
     */
    void check_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) const;

    /**
     * @brief Basic checks for input tensors
     *
     * @param port Input port
     * @param tensors Input tensors
     */
    void check_batched_tensors(const ov::Output<const ov::Node>& port,
                               const std::vector<ov::SoPtr<ov::ITensor>>& tensors) const;

    /**
     * @brief Check that all tensors are valid. Throws an exception if it's not.
     */
    void check_tensors() const override;

    /**
     * @brief Checks if the provided precision value is supported by the current backend, should throw an error
     * otherwise.
     * @param precision The precision value to be checked.
     */
    virtual void check_network_precision(const ov::element::Type_t precision) const = 0;

    /**
     * @brief Allocates a tensor on host and stores the reference inside multiple attributes.
     * @param descriptor Tensor's metadata
     * @param index The index which the allocated tensor shall use.
     * @param isInput Determines the containers in which the newly allocated tensors will be stored.
     * @param allocator If provided, the tensor uses the custom allocator instead of using the default one.
     * @param batchSize If provided, the value of the shape on the 0th axis is overriden with this value.
     * @return Pointer towards the allocated tensor
     */
    std::shared_ptr<ov::ITensor> allocate_tensor(const IODescriptor& descriptor,
                                                 const size_t index,
                                                 const bool isInput,
                                                 const ov::Allocator& allocator = {},
                                                 const std::optional<std::size_t> batchSize = std::nullopt) const;

    virtual std::shared_ptr<ov::ITensor> create_tensor(ov::element::Type type,
                                                       const ov::Shape& shape,
                                                       const ov::Allocator& allocator = {}) const;

    virtual void add_state(const IODescriptor& descriptor, const size_t tensorIndex) const;

    bool is_batched_input(size_t idx) const;

    ov::SoPtr<ov::ITensor>& get_user_input(size_t index) const;
    std::vector<ov::SoPtr<ov::ITensor>>& get_user_inputs(size_t index) const;

    // This is intel_npu::ICompiledModel pointer, but need to use OV base class because
    // ov::IInferRequest::get_compiled_model returns a refernce to shared_ptr!
    std::shared_ptr<const ov::ICompiledModel> _compiledModel;

    NetworkMetadata _metadata;

    Logger _logger;

    // In case set_tensors is called, we receive a vector with N tensors otherwise only 1 tensor is needed
    mutable std::vector<std::vector<ov::SoPtr<ov::ITensor>>> _userInputTensors;
    mutable std::vector<ov::SoPtr<ov::ITensor>> _userOutputTensors;

    mutable std::vector<ov::SoPtr<ov::IVariableState>> _variableStates;

    /**
     * @see ov::ISyncInferRequest
     */
    mutable std::unordered_map<size_t, FoundPort> _cachedPorts;

    /**
     * @see ov::ISyncInferRequest
     */
    mutable std::mutex _cacheMutex;
};

}  // namespace intel_npu
