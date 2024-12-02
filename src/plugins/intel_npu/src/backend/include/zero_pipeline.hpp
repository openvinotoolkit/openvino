// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "zero_memory.hpp"
#include "zero_profiling.hpp"

namespace intel_npu {

struct TensorData {
    void* mem;
    size_t size;
    bool levelZeroTensorCreatedLocally = true;
};

struct Pipeline {
public:
    Pipeline(const Config& config,
             const std::shared_ptr<ZeroInitStructsHolder>& initStructs,
             const std::shared_ptr<IGraph>& graph,
             zeroProfiling::ProfilingPool& profiling_pool,
             zeroProfiling::ProfilingQuery& profiling_query,
             const std::shared_ptr<zeroProfiling::NpuInferProfiling>& npu_profiling,
             const std::vector<std::vector<std::optional<TensorData>>>& inputTensorsData,
             const std::vector<std::optional<TensorData>>& outputTensorsData,
             uint32_t group_ordinal);

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    virtual ~Pipeline() = default;

    void push();
    void pull();
    void reset() const;

    void updateCommandList(const TensorData& tensorsData, uint32_t index);
    void updateCommandList(const TensorData& tensorsData, uint32_t index, size_t commandListIndex);

protected:
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

    std::shared_ptr<CommandQueue> _command_queue;
    std::vector<std::unique_ptr<CommandList>> _command_lists;
    std::vector<std::unique_ptr<Fence>> _fences;
    std::shared_ptr<EventPool> _event_pool;
    std::vector<std::shared_ptr<Event>> _events;
    bool sync_output_with_fences_ = true;
    std::shared_ptr<zeroProfiling::NpuInferProfiling> _npu_profiling;
    Logger _logger;
};

}  // namespace intel_npu
