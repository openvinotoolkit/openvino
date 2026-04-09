// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_graph_ext.h>

#include <mutex>

#include "intel_npu/common/idynamic_graph.hpp"
#include "intel_npu/common/network_metadata.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "npu_vm_execute.hpp"
#include "npu_vm_runtime_api.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {
class DynamicGraph final : public IDynamicGraph {
public:
    class Impl {
    public:
        virtual void initialize(std::optional<ov::Tensor>& blob, NetworkMetadata& metadata) = 0;
        virtual void setArgumentValue(uint32_t argi, const void* argv) = 0;
        virtual void setArgumentValueWithStrides(uint32_t argi,
                                                 const void* argv,
                                                 const std::vector<size_t>& strides) = 0;
        virtual uint64_t getNumSubgraphs() = 0;
        virtual void getBinding(GraphArguments& binding) = 0;
        virtual void predictOutputShape(std::vector<MemRefType>& inputDescriptors,
                                        std::vector<MemRefType>& outputDescriptors) = 0;
        virtual ~Impl() = default;
    };

    DynamicGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                 ov::Tensor blob,
                 bool blobAllocatedByPlugin,
                 const FilteredConfig& config);

    std::pair<uint64_t, std::optional<std::vector<uint64_t>>> export_blob(std::ostream& stream) const override;

    void set_argument_value(uint32_t argi, const void* argv) const override;

    void set_argument_value_with_strides(uint32_t id,
                                         const void* data,
                                         const std::vector<size_t>& strides) const override;

    ze_graph_handle_t get_handle() const override;

    ~DynamicGraph() override;

    const NetworkMetadata& get_metadata() const override;

    void update_network_name(std::string_view name) override;

    CommandQueueDesc get_command_queue_desc() const override;
    void set_workload_type(const ov::WorkloadType workloadType) override;
    void set_model_priority(const ov::hint::Priority modelPriority) override;

    void set_batch_size(std::size_t batch) override;

    const std::optional<std::size_t> get_batch_size() const override;

    uint32_t get_unique_id() override;
    void set_last_submitted_id(uint32_t id_index) override;
    uint32_t get_last_submitted_id() const override;

    void* get_vm_engine() const override;

    void getBinding(GraphArguments& args) override;

    uint64_t get_num_subgraphs() const override;

    void predict_output_shape(std::vector<MemRefType>& inputDescriptors,
                              std::vector<MemRefType>& outputDescriptors) override;

    std::optional<bool> is_profiling_blob() const override;

private:
    void initialize_impl(const FilteredConfig& config) override;

    bool release_blob(const FilteredConfig& config);
    std::optional<size_t> determine_batch_size();

    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;

    NetworkMetadata _metadata;

    /**
     * @brief Stores the number of subgraphs for dynamic models
     * @note the number of subgraphs will be one for static models
     */
    uint64_t _num_of_subgraphs = 1;

    mutable std::mutex _commandQueueDescMutex;
    CommandQueueDesc _commandQueueDesc;
    std::vector<std::shared_ptr<Event>> _lastSubmittedEvent;

    std::optional<ov::Tensor> _blob;

    // In the case of the import path, the blob is released after graph initialization so it can not be any longer
    // exported
    bool _blobIsReleased = false;

    uint32_t _uniqueId = 0;
    uint32_t _lastSubmittedId = 0;

    /**
     * @brief The batch size used by the corresponding model.
     * @details The attribute contains a value only if the plugin performs the batches splitting operation.
     */
    std::optional<std::size_t> _batchSize = std::nullopt;

    Logger _logger;

    std::unique_ptr<Impl> _impl;
};

}  // namespace intel_npu
