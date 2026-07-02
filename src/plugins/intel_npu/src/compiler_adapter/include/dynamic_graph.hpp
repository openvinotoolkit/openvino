// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_graph_ext.h>

#include <mutex>

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/common/network_metadata.hpp"
#include "intel_npu/utils/vm/npu_vm_runtime_api.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "intel_npu/npu_private_properties.hpp"

namespace intel_npu {
class DynamicGraph final : public IGraph {
public:
    DynamicGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                 ov::Tensor blob,
                 const FilteredConfig& config);

    std::pair<uint64_t, std::optional<std::vector<uint64_t>>> export_blob(std::ostream& stream) const override;

    void* get_handle() const override;

    bool is_dynamic() const override {
        return true;
    }

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

    std::optional<bool> is_profiling_blob() const override;

    std::optional<std::string_view> get_compatibility_descriptor() const override;

    ///这个应该放在public中吗？
    //这行应该放在 dynamic_arguments.hpp 里吗？看上去是给ececute  executeGraph的使用的
    bool _useInterpreter = true;
    bool _optimizedDynamicStridesMode = false;
    ov::intel_npu::CommandListMode _bindingCommandListMode;
    ////

private:
    /// 
    void setOptimizedDynamicStridesMode(bool mode);
    ////

    void initialize_impl(const FilteredConfig& config) override;

    bool release_blob(const FilteredConfig& config);
    std::optional<size_t> determine_batch_size();

    void initialize_engine();
    void create_execution_engine();
    void prepare_metadata();

    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;

    NetworkMetadata _metadata;

    // Preserve previous behavior: when shared common queue is disabled and a new queue is created due to a priority
    // change, keep the same workload type to avoid creating a queue with an unexpected workload.
    std::optional<ov::WorkloadType> _workloadType = std::nullopt;
    std::shared_ptr<CommandQueue> _commandQueue = nullptr;




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

    npu_vm_runtime_handle_t _engine = nullptr;
    npu_vm_runtime_properties_t _engineProperties{};
    bool _engineInitialized = false;
};

}  // namespace intel_npu
