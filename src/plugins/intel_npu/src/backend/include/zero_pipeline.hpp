// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "zero_executor.hpp"
#include "zero_memory.hpp"
#include "zero_profiling.hpp"
#include "zero_utils.hpp"
#include "zero_wrappers.hpp"

namespace intel_npu {

struct TensorData {
    void* mem;
    size_t size;
    bool levelZeroTensorCreatedLocally = true;
};

struct Pipeline {
public:
    Pipeline(const Config& config,
             const std::shared_ptr<const IExecutor>& executorPtr,
             zeroProfiling::ProfilingPool& profiling_pool,
             zeroProfiling::ProfilingQuery& profiling_query,
             std::shared_ptr<zeroProfiling::NpuInferProfiling> npu_profiling,
             const std::vector<std::vector<std::optional<TensorData>>>& inputTensorsData,
             const std::vector<std::optional<TensorData>>& outputTensorsData,
             const size_t numberOfCommandLists);

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    virtual ~Pipeline() = default;

    void push();
    void pull();
    void reset() const;

    void updateCommandList(const TensorData& tensorsData, uint32_t index);
    void updateCommandList(const TensorData& tensorsData, uint32_t index, size_t commandListIndex);

protected:
    const Config _config;
    const ZeroExecutor* _executor;
    CommandQueue& _command_queue;
    std::vector<std::unique_ptr<CommandList>> _command_lists;
    std::vector<std::unique_ptr<Fence>> _fences;
    EventPool _event_pool;
    std::vector<std::unique_ptr<Event>> _events;
    bool sync_output_with_fences_ = true;
    std::shared_ptr<zeroProfiling::NpuInferProfiling> _npu_profiling;
    Logger _logger;
};

}  // namespace intel_npu
