// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <memory>
#include <string_view>

#include "openvino/runtime/threading/itask_executor.hpp"

namespace intel_npu {

/**
 * @brief Creates a task executor with fixed-size or adaptive worker behavior.
 *
 * @param name Base thread name prefix used for worker threads.
 * @param workers Baseline number of workers to keep active.
 * @param allowWorkerGrowth If true, allows creating additional workers on demand.
 * @param idleTimeout Idle timeout after which extra adaptive workers can stop.
 * @return Shared pointer to the created task executor.
 */
std::shared_ptr<ov::threading::ITaskExecutor> make_executor(
    std::string_view name,
    size_t workers,
    bool allowWorkerGrowth = false,
    std::chrono::milliseconds idleTimeout = std::chrono::milliseconds{30'000});

}  // namespace intel_npu
