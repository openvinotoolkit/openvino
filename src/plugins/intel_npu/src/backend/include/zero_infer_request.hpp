// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include "intel_npu/common/npu.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_remote_tensor.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "zero_pipeline.hpp"
#include "zero_tensor.hpp"

namespace intel_npu {

class ZeroInferRequest final : public ov::IInferRequest {
public:
    explicit ZeroInferRequest(const std::shared_ptr<ZeroInitStructsHolder>& initStructs,
                              const std::shared_ptr<const ICompiledModel>& compiledModel,
                              const Config& config);

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
     * @brief Gets an input/output tensor for inference.
     * @note If the tensor with the specified @p port is not found, am exception is thrown.
     * @param port Port of the batched tensors to get.
     * @return Vector of batched tensors for the input port @p port or empty vector if port is output.
     */
    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& port) const override;

    /**
     * @brief Sets batched input tensors to infer
     * @param port Port of the batched input tensor.
     * @param tensors Vector of references to batched tensors. The element_type and shape of each must match.
     * @note Batched tensors for outputs is not supported.
     * @note If single element vector is provided for @p tensors param, fallback to "set_tensor" function will occur.
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
     * @brief Calls "infer_async" then "get_result"
     */
    void infer() override;

    /**
     * @brief Used for executing the inference.
     */
    void infer_async();

    /**
     * @brief Used for retrieving the prediction's result.
     */
    void get_result();

    /**
     * @brief Used for retrieving the current values of the network's variables.
     * @return Vector of each state value
     */
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

    /**
     * @brief Initializes the tensor values corresponding to the state variables.
     * @details The inital values are usually all 0s.
     */
    void initialize_states();

private:
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
            return type == Type::OUTPUT;
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

    bool is_batched_input(size_t idx) const;

    ov::SoPtr<ov::ITensor>& get_user_input(size_t index) const;
    std::vector<ov::SoPtr<ov::ITensor>>& get_user_inputs(size_t index) const;

    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    /**
     * @brief Check the received tensor and set the Level Zero tensor accordingly
     * @param tensor Reference to a tensor.
     * @param index The index corresponding to the position of the tensor inside the I/O structures.
     * @param isInput Used for identifying the structures to which the tensor belongs.
     */
    void set_tensor_data(const std::shared_ptr<ov::ITensor>& tensor, const size_t index, const bool isInput);

    /**
     * @brief Check the received remote tensor and copy it to the Level Zero tensor
     * @param tensor Reference to a tensor.
     * @param index The index corresponding to the position of the tensor inside the I/O structures.
     * @param isInput Used for identifying the structures to which the tensor belongs.
     */
    void set_remote_tensor_data(const std::shared_ptr<ZeroRemoteTensor>& tensor,
                                const size_t index,
                                const bool isInput);

    /**
     * @brief Checks if the provided precision value is supported by the current backend, should throw an error
     * otherwise.
     * @param precision The precision value to be checked.
     */
    void check_network_precision(const ov::element::Type_t precision) const;
    void create_pipeline();

    std::shared_ptr<ov::ITensor>& get_level_zero_input(size_t index, size_t tensorNo = 0) const;
    std::vector<std::shared_ptr<ov::ITensor>>& get_level_zero_inputs(size_t index) const;

    std::shared_ptr<ZeroTensor> create_tensor(ov::element::Type type,
                                              const ov::Shape& shape,
                                              const ov::Allocator& allocator = {}) const;

    void add_state(const IODescriptor& descriptor, size_t tensorIndex) const;

    void update_pipeline_if_memory_changed();
    void update_states_if_memory_changed();

    const std::shared_ptr<ZeroInitStructsHolder> _initStructs;
    const std::shared_ptr<IGraph> _graph;
    NetworkMetadata _metadata;
    // This is intel_npu::ICompiledModel pointer, but need to use OV base class because
    // ov::IInferRequest::get_compiled_model returns a refernce to shared_ptr!
    std::shared_ptr<const ov::ICompiledModel> _compiledModel;
    const Config _config;
    Logger _logger;

    const std::vector<ArgumentDescriptor>& _graphInputDescriptors;
    const std::vector<ArgumentDescriptor>& _graphOutputDescriptors;

    // A copy of each tensor is needed to maintain the original L0 memory allocation in case the user provides another
    // memory area for the tensor.
    mutable std::vector<std::vector<std::shared_ptr<ov::ITensor>>> _levelZeroInputTensors;
    mutable std::vector<std::shared_ptr<ov::ITensor>> _levelZeroOutputTensors;

    // In case set_tensors is called, we receive a vector with N tensors otherwise only 1 tensor is needed
    mutable std::vector<std::vector<ov::SoPtr<ov::ITensor>>> _userInputTensors;
    mutable std::vector<ov::SoPtr<ov::ITensor>> _userOutputTensors;

    mutable std::vector<ov::SoPtr<ov::IVariableState>> _variableStates;

    std::shared_ptr<const zeroMemory::HostMemAllocator> _inputAllocator;
    std::shared_ptr<const zeroMemory::HostMemAllocator> _outputAllocator;

    std::unique_ptr<Pipeline> _pipeline;

    bool _pipelineIsCreated = false;
    bool _dynamicBatchValueChanged = false;
    bool _externalMemoryStandardAllocationSupported = false;

    /**
     * @see ov::ISyncInferRequest
     */
    mutable std::unordered_map<size_t, FoundPort> _cachedPorts;

    /**
     * @see ov::ISyncInferRequest
     */
    mutable std::mutex _cacheMutex;
};

}  //  namespace intel_npu
