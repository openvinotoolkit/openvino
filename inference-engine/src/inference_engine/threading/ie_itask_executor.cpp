// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "threading/ie_itask_executor.hpp"

#include <future>
#include <memory>
#include <utility>
#include <vector>

namespace InferenceEngine {

void ITaskExecutor::runAndWait(const std::vector<Task>& tasks) {
    std::vector<std::packaged_task<void()>> packagedTasks;
    std::vector<std::future<void>> futures;
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        packagedTasks.emplace_back([&tasks, i] {tasks[i]();});
        futures.emplace_back(packagedTasks.back().get_future());
    }
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        run([&packagedTasks, i]{packagedTasks[i]();});
    }
    // std::future::get will rethrow exception from task.
    // We should wait all tasks before any exception is thrown.
    // So wait() and get() for each future moved to separate loops
    for (auto&& future : futures) {
        future.wait();
    }
    for (auto&& future : futures) {
        future.get();
    }
}
}  // namespace InferenceEngine
