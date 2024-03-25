// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "zero_executor.h"
#include "zero_memory.h"
#include "zero_profiling.h"
#include "zero_utils.h"
#include "zero_wrappers.h"

namespace vpux {
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

protected:
    zeroMemory::MemoryManagementUnit _deviceInputs;
    zeroMemory::MemoryManagementUnit _deviceOutputs;
};

std::unique_ptr<Pipeline> makePipeline(const std::shared_ptr<const IExecutor>& executorPtr,
                                       const Config& config,
                                       vpux::zeroProfiling::ProfilingPool& profiling_pool,
                                       vpux::zeroProfiling::ProfilingQuery& profiling_query,
                                       std::shared_ptr<vpux::zeroProfiling::VpuInferProfiling> vpu_profiling,
                                       std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>& tensors);
}  // namespace vpux
