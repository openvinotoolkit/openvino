// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/icompiled_model.hpp"
#include "intel_npu/common/igraph.hpp"
#include "intel_npu/common/network_metadata.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_tensor.hpp"
#include "zero_pipeline.hpp"

namespace intel_npu {

constexpr std::size_t SINGLE_TENSOR = 0;
constexpr bool INPUT = true;
constexpr bool OUTPUT = false;

std::optional<size_t> determine_dynamic_batch_size(const IODescriptor& desc,
                                                   const ov::PartialShape& ioShape,
                                                   const std::shared_ptr<ov::ITensor>& tensor,
                                                   const std::optional<size_t> batchSize);

void* get_tensor_data_ptr(const std::shared_ptr<ov::ITensor>& tensor);

class ZeroInferRequest : public InferRequest {
public:
    explicit ZeroInferRequest(const std::shared_ptr<ZeroInitStructsHolder>& initStructs,
                              const std::shared_ptr<const ICompiledModel>& compiledModel,
                              const Config& config);

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& port) const override;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    void set_tensors(const ov::Output<const ov::Node>& port,
                     const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

    void infer() override;
    virtual void infer_async() override;
    void get_result() override;

    const std::vector<ov::Output<const ov::Node>>& get_inputs() const override;
    const std::vector<ov::Output<const ov::Node>>& get_outputs() const override;

    const std::shared_ptr<const ov::ICompiledModel>& get_compiled_model() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

protected:
    /**
     * @see ov::ISyncInferRequest
     */
    struct FoundPort {
        size_t idx;
        enum class Type { NOT_FOUND = 0, INPUT, OUTPUT } type;

        bool found() const {
            return type != Type::NOT_FOUND;
        }
        bool is_input() const {
            return type == Type::INPUT;
        }
        bool is_output() const {
            return !is_input();
        }
    };

    /**
     * @brief Finds input or output port
     * @return structure which contains index of Input/Output or report that port wasn't found
     * @see ov::ISyncInferRequest
     */
    ZeroInferRequest::FoundPort find_port(const ov::Output<const ov::Node>& port) const;

    void setup_pipeline();
    virtual void create_pipeline_impl();

    /**
     * @brief Allocates a tensor on host and stores the reference inside multiple attributes.
     * @param index The index which the allocated tensor shall use.
     * @param isInput Determines the containers in which the newly allocated tensors will be stored.
     * @param batchSize If provided, the value of the shape on the 0th axis is overridden with this value.
     * @return Pointer towards the allocated tensor
     */
    virtual std::shared_ptr<ZeroTensor> allocate_tensor(
        const size_t index,
        const bool isInput,
        const std::optional<std::size_t>& batchSize = std::nullopt) const;

    void initialize_states();
    void add_state(const IODescriptor& descriptor, size_t tensorIndex) const;

    void update_pipeline_if_memory_changed();
    void update_states_if_memory_changed();

    virtual void sync_zero_tensor_with_graph(const ZeroInferRequest::FoundPort& foundPort,
                                             const ov::SoPtr<ov::ITensor>& tensor);
    virtual void sync_zero_tensors_with_graph(const ZeroInferRequest::FoundPort& foundPort,
                                              const std::vector<ov::SoPtr<ov::ITensor>>& tensors,
                                              const std::optional<size_t>& batchSize = std::nullopt);

    virtual void prepare_inputs();
    virtual void prepare_outputs();

    /**
     * @brief Basic checks for input/output tensor
     *
     * @param port Input/Output port
     * @param tensor Input/Output tensor
     */
    void check_tensor(const ov::Output<const ov::Node>& port,
                      const ov::SoPtr<ov::ITensor>& tensor,
                      const bool supportStrides) const;

    /**
     * @brief Basic checks for input tensors
     *
     * @param port Input port
     * @param tensors Input tensors
     */
    void check_batched_tensors(const ov::Output<const ov::Node>& port,
                               const std::vector<ov::SoPtr<ov::ITensor>>& tensors,
                               const bool supportStrides) const;

    bool is_batched_input(size_t idx) const;

    /**
     * @brief Check that all tensors are valid. Throws an exception if it's not.
     */
    void check_tensors() const override;

    ov::SoPtr<ov::ITensor>& get_user_input(size_t index) const;
    std::vector<ov::SoPtr<ov::ITensor>>& get_user_inputs(size_t index) const;

    std::shared_ptr<ZeroTensor>& get_level_zero_input(size_t index, size_t tensorNo = 0) const;
    std::vector<std::shared_ptr<ZeroTensor>>& get_level_zero_inputs(size_t index) const;

    void check_network_precision(const ov::element::Type_t precision) const;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    const std::shared_ptr<ZeroInitStructsHolder> _initStructs;

    // This is intel_npu::ICompiledModel pointer, but need to use OV base class because
    // ov::IInferRequest::get_compiled_model returns a reference to shared_ptr!
    std::shared_ptr<const ov::ICompiledModel> _compiledModel;

    const std::shared_ptr<IGraph> _graph;
    NetworkMetadata _metadata;
    const Config _config;

    // In case set_tensors is called, we receive a vector with N tensors otherwise only 1 tensor is needed
    mutable std::vector<std::vector<ov::SoPtr<ov::ITensor>>> _userInputTensors;
    mutable std::vector<ov::SoPtr<ov::ITensor>> _userOutputTensors;

    mutable std::vector<ov::SoPtr<ov::IVariableState>> _variableStates;

    // A copy of each tensor is needed to maintain the original L0 memory allocation in case the user provides another
    // memory area for the tensor.
    mutable std::vector<std::vector<std::shared_ptr<ZeroTensor>>> _levelZeroInputTensors;
    mutable std::vector<std::shared_ptr<ZeroTensor>> _levelZeroOutputTensors;

    std::unique_ptr<IPipeline> _pipeline;
    bool _pipelineIsCreated = false;
    bool _dynamicBatchValueChanged = false;

    Logger _logger;

    /**
     * @see ov::ISyncInferRequest
     */
    mutable std::unordered_map<size_t, ZeroInferRequest::FoundPort> _cachedPorts;

    /**
     * @see ov::ISyncInferRequest
     */
    mutable std::mutex _cacheMutex;

    bool _isShapeTensorPresent = false;
};

}  //  namespace intel_npu
