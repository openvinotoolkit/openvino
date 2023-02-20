// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "threading/ie_executor_manager.hpp"

#include "ie_parallel.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"
#include "threading/ie_cpu_streams_executor.hpp"
#include "threading/ie_istreams_executor.hpp"
#include "threading/ie_itask_executor.hpp"
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

namespace {

class TaskExecutorWrapper : public InferenceEngine::ITaskExecutor {
public:
    TaskExecutorWrapper(const std::shared_ptr<ov::ITaskExecutor>& task_executor) : m_task_executor(task_executor) {}

    void run(InferenceEngine::Task task) override {
        m_task_executor->run(task);
    }

    void runAndWait(const std::vector<InferenceEngine::Task>& tasks) override {
        m_task_executor->run_and_wait(tasks);
    }

private:
    std::shared_ptr<ov::ITaskExecutor> m_task_executor;
};

class StreamsExecutorWrapper : public InferenceEngine::IStreamsExecutor {
public:
    StreamsExecutorWrapper(const std::shared_ptr<ov::IStreamsExecutor>& streams_executor)
        : m_streams_executor(streams_executor) {}

    int GetStreamId() override {
        return m_streams_executor->get_stream_id();
    }

    /**
     * @brief Return the id of current NUMA Node
     * @return `ID` of current NUMA Node, or throws exceptions if called not from stream thread
     */
    int GetNumaNodeId() override {
        return m_streams_executor->get_numa_node_id();
    }

    /**
     * @brief Execute the task in the current thread using streams executor configuration and constraints
     * @param task A task to start
     */
    void Execute(InferenceEngine::Task task) override {
        m_streams_executor->execute(task);
    }

    void run(InferenceEngine::Task task) override {
        m_streams_executor->run(task);
    }

    void runAndWait(const std::vector<InferenceEngine::Task>& tasks) override {
        m_streams_executor->run_and_wait(tasks);
    }

private:
    std::shared_ptr<ov::IStreamsExecutor> m_streams_executor;
};

ov::IStreamsExecutor::Config convert_config(const InferenceEngine::IStreamsExecutor::Config& config) {
    ov::IStreamsExecutor::Config ov_config(
        config._name,
        config._streams,
        config._threadsPerStream,
        static_cast<ov::IStreamsExecutor::ThreadBindingType>(config._threadBindingType),
        config._threadBindingStep,
        config._threadBindingOffset,
        config._threads,
        static_cast<ov::IStreamsExecutor::Config::PreferredCoreType>(config._threadPreferredCoreType));
    return ov_config;
}

}  // namespace

namespace InferenceEngine {

ExecutorManager::ExecutorManager(const std::shared_ptr<ov::ExecutorManager>& manager) : m_manager(manager) {}

ExecutorManager::~ExecutorManager() = default;

void ExecutorManager::setTbbFlag(bool flag) {
    m_manager->set_tbb_flag(flag);
}

bool ExecutorManager::getTbbFlag() {
    return m_manager->get_tbb_flag();
}

ITaskExecutor::Ptr ExecutorManager::getExecutor(const std::string& id) {
    return std::make_shared<TaskExecutorWrapper>(m_manager->get_executor(id));
}

IStreamsExecutor::Ptr ExecutorManager::getIdleCPUStreamsExecutor(const IStreamsExecutor::Config& config) {
    return std::make_shared<StreamsExecutorWrapper>(m_manager->get_idle_cpu_streams_executor(convert_config(config)));
}

size_t ExecutorManager::getExecutorsNumber() const {
    return m_manager->get_executors_number();
}

size_t ExecutorManager::getIdleCPUStreamsExecutorsNumber() const {
    return m_manager->get_idle_cpu_streams_executors_number();
}

void ExecutorManager::clear(const std::string& id) {
    m_manager->clear(id);
}

namespace {

class ExecutorManagerHolder {
    std::mutex _mutex;
    std::weak_ptr<ExecutorManager> _manager;

public:
    ExecutorManagerHolder(const ExecutorManagerHolder&) = delete;
    ExecutorManagerHolder& operator=(const ExecutorManagerHolder&) = delete;

    ExecutorManagerHolder() = default;

    ExecutorManager::Ptr get() {
        std::lock_guard<std::mutex> lock(_mutex);
        auto manager = _manager.lock();
        if (!manager) {
            _manager = manager = std::make_shared<ExecutorManager>(ov::executor_manager());
        }
        return manager;
    }
};

}  // namespace

ExecutorManager::Ptr executorManager() {
    static ExecutorManagerHolder executorManagerHolder;
    return executorManagerHolder.get();
}

ExecutorManager* ExecutorManager::getInstance() {
    static auto ptr = executorManager().get();
    return ptr;
}

}  // namespace InferenceEngine
