// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/threading/executor_manager.hpp"

#include "ie_parallel.hpp"
#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#include "threading/ie_cpu_streams_executor.hpp"
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
#    if (TBB_INTERFACE_VERSION < 12000)
#        include <tbb/task_scheduler_init.h>
#    else
#        include <oneapi/tbb/global_control.h>
#    endif
#endif

#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace ov {

namespace {
class ExecutorManagerImpl : public ExecutorManager {
public:
    ~ExecutorManagerImpl();
    std::shared_ptr<ITaskExecutor> get_executor(const std::string& id) override;
    std::shared_ptr<IStreamsExecutor> get_idle_cpu_streams_executor(
        const IStreamsExecutor::Configuration& config) override;
    size_t get_executors_number() const override;
    size_t get_idle_cpu_streams_executors_number() const override;
    void clear(const std::string& id = {}) override;
    void set_tbb_flag(bool flag) override;
    bool get_tbb_flag() override;

private:
    void reset_tbb();
    std::unordered_map<std::string, std::shared_ptr<ITaskExecutor>> executors;
    std::vector<std::pair<IStreamsExecutor::Configuration, std::shared_ptr<IStreamsExecutor>>> cpuStreamsExecutors;
    mutable std::mutex streamExecutorMutex;
    mutable std::mutex taskExecutorMutex;
    bool tbbTerminateFlag = false;
    mutable std::mutex tbbMutex;
    bool tbbThreadsCreated = false;
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
#    if (TBB_INTERFACE_VERSION < 12000)
    std::shared_ptr<tbb::task_scheduler_init> tbbTaskScheduler = nullptr;
#    else
    std::shared_ptr<oneapi::tbb::task_scheduler_handle> tbbTaskScheduler = nullptr;
#    endif
#endif
};

}  // namespace

ExecutorManagerImpl::~ExecutorManagerImpl() {
    reset_tbb();
}

void ExecutorManagerImpl::set_tbb_flag(bool flag) {
    std::lock_guard<std::mutex> guard(tbbMutex);
    tbbTerminateFlag = flag;
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
    if (tbbTerminateFlag) {
        if (!tbbTaskScheduler) {
#    if (TBB_INTERFACE_VERSION < 12000)
            tbbTaskScheduler = std::make_shared<tbb::task_scheduler_init>();
#    elif (TBB_INTERFACE_VERSION < 12060)
            tbbTaskScheduler =
                std::make_shared<oneapi::tbb::task_scheduler_handle>(oneapi::tbb::task_scheduler_handle::get());
#    else
            tbbTaskScheduler = std::make_shared<oneapi::tbb::task_scheduler_handle>(tbb::attach{});
#    endif
        }
    } else {
        tbbTaskScheduler = nullptr;
    }
#endif
}

bool ExecutorManagerImpl::get_tbb_flag() {
    std::lock_guard<std::mutex> guard(tbbMutex);
    return tbbTerminateFlag;
}

void ExecutorManagerImpl::reset_tbb() {
    std::lock_guard<std::mutex> guard(tbbMutex);
    if (tbbTerminateFlag) {
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
        if (tbbTaskScheduler && tbbThreadsCreated) {
#    if (TBB_INTERFACE_VERSION < 12000)
            tbbTaskScheduler->terminate();
#    else
            tbb::finalize(*tbbTaskScheduler, std::nothrow);
#    endif
        }
        tbbThreadsCreated = false;
        tbbTaskScheduler = nullptr;
#endif
        tbbTerminateFlag = false;
    }
}

std::shared_ptr<ITaskExecutor> ExecutorManagerImpl::get_executor(const std::string& id) {
    std::lock_guard<std::mutex> guard(taskExecutorMutex);
    auto foundEntry = executors.find(id);
    if (foundEntry == executors.end()) {
        auto newExec = std::make_shared<ov::CPUStreamsExecutor>(IStreamsExecutor::Configuration{id});
        tbbThreadsCreated = true;
        executors[id] = newExec;
        return newExec;
    }
    return foundEntry->second;
}

std::shared_ptr<IStreamsExecutor> ExecutorManagerImpl::get_idle_cpu_streams_executor(
    const IStreamsExecutor::Configuration& config) {
    std::lock_guard<std::mutex> guard(streamExecutorMutex);
    for (const auto& it : cpuStreamsExecutors) {
        const auto& executor = it.second;
        if (executor.use_count() != 1)
            continue;

        const auto& executorConfig = it.first;
        if (executorConfig._name == config._name && executorConfig._streams == config._streams &&
            executorConfig._threadsPerStream == config._threadsPerStream &&
            executorConfig._threadBindingType == config._threadBindingType &&
            executorConfig._threadBindingStep == config._threadBindingStep &&
            executorConfig._threadBindingOffset == config._threadBindingOffset)
            if (executorConfig._threadBindingType != IStreamsExecutor::ThreadBindingType::HYBRID_AWARE ||
                executorConfig._threadPreferredCoreType == config._threadPreferredCoreType)
                return executor;
    }
    auto newExec = std::make_shared<CPUStreamsExecutor>(config);
    tbbThreadsCreated = true;
    cpuStreamsExecutors.emplace_back(std::make_pair(config, newExec));
    return newExec;
}

size_t ExecutorManagerImpl::get_executors_number() const {
    std::lock_guard<std::mutex> guard(taskExecutorMutex);
    return executors.size();
}

size_t ExecutorManagerImpl::get_idle_cpu_streams_executors_number() const {
    std::lock_guard<std::mutex> guard(streamExecutorMutex);
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
            std::remove_if(
                cpuStreamsExecutors.begin(),
                cpuStreamsExecutors.end(),
                [&](const std::pair<IStreamsExecutor::Configuration, std::shared_ptr<IStreamsExecutor>>& it) {
                    return it.first._name == id;
                }),
            cpuStreamsExecutors.end());
    }
}

namespace {

class ExecutorManagerHolder {
    std::mutex _mutex;
    std::weak_ptr<ExecutorManager> _manager;

public:
    ExecutorManagerHolder(const ExecutorManagerHolder&) = delete;
    ExecutorManagerHolder& operator=(const ExecutorManagerHolder&) = delete;

    ExecutorManagerHolder() = default;

    std::shared_ptr<ExecutorManager> get() {
        std::lock_guard<std::mutex> lock(_mutex);
        auto manager = _manager.lock();
        if (!manager) {
            _manager = manager = std::make_shared<ExecutorManagerImpl>();
        }
        return manager;
    }
};

}  // namespace

std::shared_ptr<ExecutorManager> executor_manager() {
    static ExecutorManagerHolder executorManagerHolder;
    return executorManagerHolder.get();
}

}  // namespace ov
