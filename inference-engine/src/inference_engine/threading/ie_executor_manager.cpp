// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <utility>

#include "threading/ie_executor_manager.hpp"
#include "threading/ie_cpu_streams_executor.hpp"

namespace InferenceEngine {

ITaskExecutor::Ptr ExecutorManagerImpl::getExecutor(std::string id) {
    auto foundEntry = executors.find(id);
    if (foundEntry == executors.end()) {
        auto newExec = std::make_shared<CPUStreamsExecutor>(IStreamsExecutor::Config{id});
        executors[id] = newExec;
        return newExec;
    }
    return foundEntry->second;
}

IStreamsExecutor::Ptr ExecutorManagerImpl::getIdleCPUStreamsExecutor(const IStreamsExecutor::Config& config) {
    for (const auto& it : cpuStreamsExecutors) {
        const auto& executor = it.second;
        if (executor.use_count() != 1)
            continue;

        const auto& executorConfig = it.first;
        if (executorConfig._name == config._name &&
            executorConfig._streams == config._streams &&
            executorConfig._threadsPerStream == config._threadsPerStream &&
            executorConfig._threadBindingType == config._threadBindingType &&
            executorConfig._threadBindingStep == config._threadBindingStep &&
            executorConfig._threadBindingOffset == config._threadBindingOffset)
            return executor;
    }
    auto newExec = std::make_shared<CPUStreamsExecutor>(config);
    cpuStreamsExecutors.emplace_back(std::make_pair(config, newExec));
    return newExec;
}

// for tests purposes
size_t ExecutorManagerImpl::getExecutorsNumber() {
    return executors.size();
}

// for tests purposes
size_t ExecutorManagerImpl::getIdleCPUStreamsExecutorsNumber() {
    return cpuStreamsExecutors.size();
}

void ExecutorManagerImpl::clear(const std::string& id) {
    if (id.empty()) {
        executors.clear();
        cpuStreamsExecutors.clear();
    } else {
        executors.erase(id);
        cpuStreamsExecutors.erase(
            std::remove_if(cpuStreamsExecutors.begin(), cpuStreamsExecutors.end(),
                           [&](const std::pair<IStreamsExecutor::Config, IStreamsExecutor::Ptr>& it) {
                              return it.first._name == id;
                           }),
            cpuStreamsExecutors.end());
    }
}

ExecutorManager* ExecutorManager::_instance = nullptr;

ITaskExecutor::Ptr ExecutorManager::getExecutor(std::string id) {
    return _impl.getExecutor(id);
}

size_t ExecutorManager::getExecutorsNumber() {
    return _impl.getExecutorsNumber();
}

size_t ExecutorManager::getIdleCPUStreamsExecutorsNumber() {
    return _impl.getIdleCPUStreamsExecutorsNumber();
}

void ExecutorManager::clear(const std::string& id) {
    _impl.clear(id);
}

IStreamsExecutor::Ptr ExecutorManager::getIdleCPUStreamsExecutor(const IStreamsExecutor::Config& config) {
    return _impl.getIdleCPUStreamsExecutor(config);
}

}  // namespace InferenceEngine
