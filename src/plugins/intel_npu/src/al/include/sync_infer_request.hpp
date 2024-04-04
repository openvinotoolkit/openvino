// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/al/icompiled_model.hpp"
#include "intel_npu/al/icompiler.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "variable_state.hpp"

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
    explicit SyncInferRequest(const std::shared_ptr<const ICompiledModel>& compiledModel);

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
     * @brief Currently there is no support implemented for batches of tensors, thus this call is a simple redirection
     * to the "get_tensor" one.
     */
    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& port) const override;

    /**
     * @brief Currently there is no support implemented for batches of tensors, thus this call is a simple redirection
     * to the "set_tensor" one.
     */
    void set_tensors(const ov::Output<const ov::Node>& port,
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

    /**
     * @return The state tensors accessible by their names.
     */
    std::unordered_map<std::string, std::shared_ptr<VariableState>>& get_variable_states() {
        return _variableStates;
    }

    /**
     * @return The names used by the inputs in the order registered inside the model.
     */
    std::vector<std::string> get_input_names() {
        return _metadata.inputNames;
    }

    /**
     * @return The names used by the outputs in the order registered inside the model.
     */
    std::vector<std::string> get_output_names() {
        return _metadata.outputNames;
    }

    /**
     * @return The names used by the state variables in the order registered inside the model.
     */
    std::vector<std::string> get_state_names() {
        return _metadata.stateNames;
    }

    /**
     * @return The names used by the shape variables in the order registered inside the model.
     */
    std::vector<std::string> get_shape_names() {
        return _metadata.shapeNames;
    }

    /**
     * @return A map holding references towards all tensors used by the current inference request object.
     */
    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>& get_all_tensors() {
        return _allTensors;
    }

    /**
     * @return A map holding references towards all shapes tensors used by the current inference request object.
     */
    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>& get_shapes_tensors() {
        return _shapesTensors;
    }

protected:
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
     * @brief Checks if the provided precision value is supported by the current backend, should throw an error
     * otherwise.
     * @param precision The precision value to be checked.
     */
    virtual void check_network_precision(const ov::element::Type_t precision) = 0;

    /**
     * @brief Indicates a kind of provided tensor. Marks special tensors, used for internal implementation
     */
    enum class TensorType { InputOrOutput, Shape, State };

    /**
     * @brief Allocates a tensor on host and stores the reference inside the "_allTensors" attribute. If a buffer
     * address is provided, then the tensor is built upon it and no additional data buffer is allocated.
     * @param tensorName The name by which the tensor shall be identified
     * @param descriptor Tensor's metadata
     * @param isState If true, the tensor shall also be stored inside the state variables map. In this case, adding the
     * tensor to this structure would be required in order to correctly answer the state queries.
     * @param allocator If provided, the tensor uses the custom allocator instead of using the default one.
     */
    void allocate_tensor(std::string tensorName,
                         const IONodeDescriptor& descriptor,
                         TensorType tensorType = TensorType::InputOrOutput,
                         const ov::Allocator& allocator = {});

    // Mutable to return reference to ov::Tensor
    mutable std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> _allTensors;
    mutable std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> _shapesTensors;
    // A copy of each tensor is needed to maintain the original L0 memory allocation in case the user provides another
    // memory area for the tensor.
    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> _copyAllTensors;

    std::unordered_map<std::string, std::shared_ptr<VariableState>> _variableStates;

    // This is intel_npu::ICompiledModel pointer, but need to use OV base class because
    // ov::IInferRequest::get_compiled_model returns a refernce to shared_ptr!
    std::shared_ptr<const ov::ICompiledModel> _compiledModel;

    NetworkMetadata _metadata;

    // Stored in order to avoid additional processing when launching inferences
    std::vector<std::string> _inputAndStateInputNames;
    std::vector<std::string> _outputAndStateOutputNames;

    std::unordered_map<std::string, std::string> _nodeNameToLegacyName;
};

}  // namespace intel_npu
