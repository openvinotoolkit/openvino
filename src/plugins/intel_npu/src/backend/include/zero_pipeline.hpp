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
    bool changed = false;
};

struct Pipeline {
public:
    Pipeline() = default;
    Pipeline(const Pipeline&) = delete;
    Pipeline(Pipeline&&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline& operator=(Pipeline&&) = delete;
    virtual ~Pipeline() = default;

    virtual void push() = 0;
    virtual void pull() = 0;
    virtual void reset() const = 0;

    virtual void updateCommandList(std::vector<std::optional<TensorData>>& inputTensorsData,
                                   std::vector<std::optional<TensorData>>& outputTensorsData) = 0;

protected:
    zeroMemory::MemoryManagementUnit _deviceInputs;
    zeroMemory::MemoryManagementUnit _deviceOutputs;
};

std::unique_ptr<Pipeline> makePipeline(const std::shared_ptr<const IExecutor>& executorPtr,
                                       const Config& config,
                                       zeroProfiling::ProfilingPool& profiling_pool,
                                       zeroProfiling::ProfilingQuery& profiling_query,
                                       std::shared_ptr<zeroProfiling::NpuInferProfiling> npu_profiling,
                                       const std::vector<std::optional<TensorData>>& inputTensorsData,
                                       const std::vector<std::optional<TensorData>>& outputTensorsData,
                                       const size_t numberOfCommandLists);
}  // namespace intel_npu
