// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include "cpp_interfaces/ie_executor_manager.hpp"
#include "cpp_interfaces/ie_task_executor.hpp"

namespace InferenceEngine {

ITaskExecutor::Ptr ExecutorManagerImpl::getExecutor(std::string id) {
    auto foundEntry = executors.find(id);
    if (foundEntry == executors.end()) {
        auto newExec = std::make_shared<TaskExecutor>(id);
        executors[id] = newExec;
        return newExec;
    }
    return foundEntry->second;
}

// for tests purposes
size_t ExecutorManagerImpl::getExecutorsNumber() {
    return executors.size();
}

void ExecutorManagerImpl::clear() {
    executors.clear();
}

ExecutorManager *ExecutorManager::_instance = nullptr;

ITaskExecutor::Ptr ExecutorManager::getExecutor(std::string id) {
    return _impl.getExecutor(id);
}

size_t ExecutorManager::getExecutorsNumber() {
    return _impl.getExecutorsNumber();
}

void ExecutorManager::clear() {
    _impl.clear();
}

}  // namespace InferenceEngine
