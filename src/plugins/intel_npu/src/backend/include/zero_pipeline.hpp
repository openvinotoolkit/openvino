// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "zero_memory.hpp"
#include "zero_profiling.hpp"
#include "zero_tensor.hpp"

namespace intel_npu {

struct Pipeline {
public:
    Pipeline(const Config& config,
             const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
             const std::shared_ptr<IGraph>& graph,
             zeroProfiling::ProfilingPool& profiling_pool,
             zeroProfiling::ProfilingQuery& profiling_query,
             const std::shared_ptr<zeroProfiling::NpuInferProfiling>& npu_profiling,
             const std::vector<std::vector<std::shared_ptr<ov::ITensor>>>& input_tensors,
             const std::vector<std::shared_ptr<ov::ITensor>>& output_tensors,
             uint32_t group_ordinal);

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    ~Pipeline();

    void push();
    void pull();
    void reset() const;

    void updateCommandList(uint32_t arg_index, const void* arg_data, size_t byte_size);
    void updateCommandListIndex(uint32_t arg_index, const void* arg_data, size_t command_list_index);

    void closeCommandList();
    void closeCommandListIndex(size_t command_list_index);

protected:
    void getCommandQueue();

    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    std::shared_ptr<IGraph> _graph;
    const Config _config;
    const uint32_t _id;

    /**
     * @brief Indicates how many command lists will be used inside the pipeline.
     * @details Leveraging multiple command lists implies distributing the input/output buffers accross the batch axis
     * between these lists.
     *
     * If batching is handled on compiler's side then a single command list shall be used, we don't do any
     * specific operation inside the plugin in this case.
     */
    size_t _number_of_command_lists;

    CommandQueueFactory _command_queue_factory;
    std::shared_ptr<CommandQueue> _command_queue;
    std::vector<std::unique_ptr<CommandList>> _command_lists;
    std::vector<std::unique_ptr<Fence>> _fences;
    std::shared_ptr<EventPool> _event_pool;
    std::vector<std::shared_ptr<Event>> _events;
    bool _sync_output_with_fences = true;
    std::shared_ptr<zeroProfiling::NpuInferProfiling> _npu_profiling;
    Logger _logger;

    uint32_t _group_ordinal;
    bool _fences_are_created = false;
    std::mutex _mutex;
    bool _turbo = false;
    ze_command_queue_priority_t _ze_queue_priority;
    std::optional<ze_command_queue_workload_type_t> _ze_workload_type = std::nullopt;
};

}  // namespace intel_npu
