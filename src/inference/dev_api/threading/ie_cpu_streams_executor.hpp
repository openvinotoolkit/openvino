// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file ie_cpu_streams_executor.hpp
 * @brief A header file for Inference Engine CPU-Streams-based Executor implementation.
 */

#pragma once

#include <memory>
#include <string>

#include "threading/ie_istreams_executor.hpp"

namespace InferenceEngine {
/**
 * @class CPUStreamsExecutor
 * @ingroup ie_dev_api_threading
 * @brief CPU Streams executor implementation. The executor splits the CPU into groups of threads,
 *        that can be pinned to cores or NUMA nodes.
 *        It uses custom threads to pull tasks from single queue.
 */
class INFERENCE_ENGINE_API_CLASS(CPUStreamsExecutor) : public IStreamsExecutor {
public:
    /**
     * @brief A shared pointer to a CPUStreamsExecutor object
     */
    using Ptr = std::shared_ptr<CPUStreamsExecutor>;

    /**
     * @brief Constructor
     * @param config Stream executor parameters
     */
    explicit CPUStreamsExecutor(const IStreamsExecutor::Config& config = {});

    /**
     * @brief A class destructor
     */
    ~CPUStreamsExecutor() override;

    void run(Task task) override;

    void Execute(Task task) override;

    int GetStreamId() override;

    int GetNumaNodeId() override;

    int GetSocketId() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

}  // namespace InferenceEngine
