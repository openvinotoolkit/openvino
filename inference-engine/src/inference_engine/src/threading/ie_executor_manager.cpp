// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <utility>

#include "threading/ie_executor_manager.hpp"
#include "threading/ie_cpu_streams_executor.hpp"

namespace InferenceEngine {

ITaskExecutor::Ptr ExecutorManagerImpl::getExecutor(std::string id) {
    std::lock_guard<std::mutex> guard(taskExecutorMutex);
    auto foundEntry = executors.find(id);
    if (foundEntry == executors.end()) {
        auto newExec = std::make_shared<CPUStreamsExecutor>(IStreamsExecutor::Config{id});
        executors[id] = newExec;
        return newExec;
    }
    return foundEntry->second;
}

IStreamsExecutor::Ptr ExecutorManagerImpl::getIdleCPUStreamsExecutor(const IStreamsExecutor::Config& config) {
    std::lock_guard<std::mutex> guard(streamExecutorMutex);
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
            if (executorConfig._threadBindingType != IStreamsExecutor::ThreadBindingType::HYBRID_AWARE
                 || executorConfig._threadPreferredCoreType == config._threadPreferredCoreType)
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
    std::lock_guard<std::mutex> stream_guard(streamExecutorMutex);
    std::lock_guard<std::mutex> task_guard(taskExecutorMutex);
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

std::mutex ExecutorManager::_mutex;
ExecutorManager* ExecutorManager::_instance = nullptr;

ExecutorManager* ExecutorManager::getInstance() {
    /*
     * 1) We do not use singleton implementation via STATIC LOCAL object like
     *
     *   getInstance() {
     *       static ExecutorManager _instance;
     *       return &instance;
     *   }
     *
     * Because of problem with destruction order on program exit.
     * Some IE classes like MKLDNN::Engine use this singleton in destructor.
     * But they has no direct dependency from c++ runtime point of view and
     * it's possible that _instance local static variable  will be destroyed
     * before MKLDNN::~Engine call. Any further manipulation with destroyed
     * object will lead to exception or crashes.
     *
     * 2) We do not use singleton implementation via STATIC object like:
     *
     *   ExecutorManager ExecutorManager::_instance;
     *   getInstance() {
     *       return &instance;
     *   }
     *
     * Because of problem with double destruction. In some test cases we use
     * double link with IE module via static and dynamic version. Both modules
     * have static object with same export name and it leads to double construction
     * and double destruction of that object. For some c++ compilers (ex gcc 5.4)
     * it lead to crash with "double free".
     *
     * That's why we use manual allocation of singleton instance on heap.
     */
    std::lock_guard<std::mutex> guard(_mutex);
    if (_instance == nullptr) {
        _instance = new ExecutorManager();
    }
    return _instance;
}

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
