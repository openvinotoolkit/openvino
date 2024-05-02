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
struct Pipeline {
public:
    Pipeline() = default;
    Pipeline(const Pipeline&) = delete;
    Pipeline(Pipeline&&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline& operator=(Pipeline&&) = delete;
    virtual ~Pipeline() = default;

    virtual void push(size_t batch_index) = 0;
    virtual void pull(size_t batch_index) = 0;
    virtual void reset(size_t batch_index) const = 0;

protected:
    zeroMemory::MemoryManagementUnit _deviceInputs;
    zeroMemory::MemoryManagementUnit _deviceOutputs;
};

std::unique_ptr<Pipeline> makePipeline(const std::shared_ptr<const IExecutor>& executorPtr,
                                       const Config& config,
                                       zeroProfiling::ProfilingPool& profiling_pool,
                                       zeroProfiling::ProfilingQuery& profiling_query,
                                       std::shared_ptr<zeroProfiling::NpuInferProfiling> npu_profiling,
                                       const std::vector<std::shared_ptr<ov::ITensor>>& inputTensors,
                                       const std::vector<std::shared_ptr<ov::ITensor>>& outputTensors,
                                       const size_t batch_size);
}  // namespace intel_npu
