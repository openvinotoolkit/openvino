// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/threading/executor_manager.hpp"

#include "openvino/core/parallel.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO
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
namespace threading {
namespace {
class ExecutorManagerImpl : public ExecutorManager {
public:
    ~ExecutorManagerImpl();
    std::shared_ptr<ov::threading::ITaskExecutor> get_executor(const std::string& id) override;
    std::shared_ptr<ov::threading::IStreamsExecutor> get_idle_cpu_streams_executor(
        const ov::threading::IStreamsExecutor::Config& config) override;
    size_t get_executors_number() const override;
    size_t get_idle_cpu_streams_executors_number() const override;
    void clear(const std::string& id = {}) override;
    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;
    void execute_task_by_streams_executor(ov::hint::SchedulingCoreType core_type, ov::threading::Task task) override;

private:
    void reset_tbb();

    std::unordered_map<std::string, std::shared_ptr<ov::threading::ITaskExecutor>> executors;
    std::vector<std::pair<ov::threading::IStreamsExecutor::Config, std::shared_ptr<ov::threading::IStreamsExecutor>>>
        cpuStreamsExecutors;
    mutable std::mutex streamExecutorMutex;
    mutable std::mutex taskExecutorMutex;
    bool tbbTerminateFlag = false;
    mutable std::mutex global_mutex;
    bool tbbThreadsCreated = false;
#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO
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

void ExecutorManagerImpl::set_property(const ov::AnyMap& properties) {
    std::lock_guard<std::mutex> guard(global_mutex);
    for (const auto& it : properties) {
        if (it.first == ov::force_tbb_terminate.name()) {
            tbbTerminateFlag = it.second.as<bool>();
#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO
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
    }
}
ov::Any ExecutorManagerImpl::get_property(const std::string& name) const {
    std::lock_guard<std::mutex> guard(global_mutex);
    if (name == ov::force_tbb_terminate.name()) {
        return tbbTerminateFlag;
    }
    OPENVINO_THROW("Property ", name, " is not supported.");
}

void ExecutorManagerImpl::reset_tbb() {
    std::lock_guard<std::mutex> guard(global_mutex);
    if (tbbTerminateFlag) {
#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO
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

std::shared_ptr<ov::threading::ITaskExecutor> ExecutorManagerImpl::get_executor(const std::string& id) {
    std::lock_guard<std::mutex> guard(taskExecutorMutex);
    auto foundEntry = executors.find(id);
    if (foundEntry == executors.end()) {
        auto newExec = std::make_shared<ov::threading::CPUStreamsExecutor>(ov::threading::IStreamsExecutor::Config{id});
        tbbThreadsCreated = true;
        executors[id] = newExec;
        return newExec;
    }
    return foundEntry->second;
}

std::shared_ptr<ov::threading::IStreamsExecutor> ExecutorManagerImpl::get_idle_cpu_streams_executor(
    const ov::threading::IStreamsExecutor::Config& config) {
    std::lock_guard<std::mutex> guard(streamExecutorMutex);
    for (auto& it : cpuStreamsExecutors) {
        const auto& executor = it.second;
        if (executor.use_count() != 1)
            continue;

        auto& executorConfig = it.first;
        if (executorConfig == config)
            return executor;
    }
    auto newExec = std::make_shared<ov::threading::CPUStreamsExecutor>(config);
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
        cpuStreamsExecutors.erase(std::remove_if(cpuStreamsExecutors.begin(),
                                                 cpuStreamsExecutors.end(),
                                                 [&](std::pair<ov::threading::IStreamsExecutor::Config,
                                                               std::shared_ptr<ov::threading::IStreamsExecutor>>& it) {
                                                     return it.first.get_name() == id;
                                                 }),
                                  cpuStreamsExecutors.end());
    }
}

void ExecutorManagerImpl::execute_task_by_streams_executor(ov::hint::SchedulingCoreType core_type,
                                                           ov::threading::Task task) {
    ov::threading::IStreamsExecutor::Config streamsConfig("StreamsExecutor", 1, 1, core_type);
    if (!streamsConfig.get_streams_info_table().empty()) {
        auto taskExecutor = std::make_shared<ov::threading::CPUStreamsExecutor>(streamsConfig);
        std::vector<Task> tasks{std::move(task)};
        taskExecutor->run_and_wait(tasks);
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

    std::shared_ptr<ov::threading::ExecutorManager> get() {
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

}  // namespace threading
}  // namespace ov
