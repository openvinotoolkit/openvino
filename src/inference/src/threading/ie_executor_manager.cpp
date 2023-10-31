// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "threading/ie_executor_manager.hpp"

#include "ie_parallel.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"
#include "threading/ie_cpu_streams_executor.hpp"
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

namespace InferenceEngine {
namespace {
class ExecutorManagerImpl : public ExecutorManager {
public:
    ExecutorManagerImpl(const std::shared_ptr<ov::threading::ExecutorManager>& manager);
    ITaskExecutor::Ptr getExecutor(const std::string& id) override;
    IStreamsExecutor::Ptr getIdleCPUStreamsExecutor(const IStreamsExecutor::Config& config) override;
    size_t getExecutorsNumber() const override;
    size_t getIdleCPUStreamsExecutorsNumber() const override;
    void clear(const std::string& id = {}) override;
    void setTbbFlag(bool flag) override;
    bool getTbbFlag() override;

private:
    std::shared_ptr<ov::threading::ExecutorManager> m_manager;
    std::shared_ptr<ov::threading::ExecutorManager> get_ov_manager() const override {
        return m_manager;
    }
};

class TaskExecutorWrapper : public ITaskExecutor {
    std::shared_ptr<ov::threading::ITaskExecutor> m_executor;

public:
    TaskExecutorWrapper(const std::shared_ptr<ov::threading::ITaskExecutor>& executor) : m_executor(executor) {}
    void run(Task task) override {
        m_executor->run(task);
    }

    void runAndWait(const std::vector<Task>& tasks) override {
        m_executor->run_and_wait(tasks);
    }
};

class StreamsExecutorWrapper : public IStreamsExecutor {
    std::shared_ptr<ov::threading::IStreamsExecutor> m_executor;

public:
    StreamsExecutorWrapper(const std::shared_ptr<ov::threading::IStreamsExecutor>& executor) : m_executor(executor) {}
    void run(Task task) override {
        m_executor->run(task);
    }

    void runAndWait(const std::vector<Task>& tasks) override {
        m_executor->run_and_wait(tasks);
    }
    int GetStreamId() override {
        return m_executor->get_stream_id();
    }

    int GetNumaNodeId() override {
        return m_executor->get_numa_node_id();
    }

    int GetSocketId() override {
        return m_executor->get_socket_id();
    }

    std::vector<int> GetCoresMtSockets() override {
        return m_executor->get_cores_mt_sockets();
    }

    void Execute(Task task) override {
        m_executor->execute(task);
    }

    void run_id(Task task, int id = -1) override {
        m_executor->run_id(task, id);
    }
};

}  // namespace

ExecutorManagerImpl::ExecutorManagerImpl(const std::shared_ptr<ov::threading::ExecutorManager>& manager)
    : m_manager(manager) {}

void ExecutorManagerImpl::setTbbFlag(bool flag) {
    m_manager->set_property({{ov::force_tbb_terminate.name(), flag}});
}

bool ExecutorManagerImpl::getTbbFlag() {
    return m_manager->get_property(ov::force_tbb_terminate.name()).as<bool>();
}

ITaskExecutor::Ptr ExecutorManagerImpl::getExecutor(const std::string& id) {
    return std::make_shared<TaskExecutorWrapper>(m_manager->get_executor(id));
}

IStreamsExecutor::Ptr ExecutorManagerImpl::getIdleCPUStreamsExecutor(const IStreamsExecutor::Config& config) {
    return std::make_shared<StreamsExecutorWrapper>(m_manager->get_idle_cpu_streams_executor(config));
}

size_t ExecutorManagerImpl::getExecutorsNumber() const {
    return m_manager->get_executors_number();
}

size_t ExecutorManagerImpl::getIdleCPUStreamsExecutorsNumber() const {
    return m_manager->get_idle_cpu_streams_executors_number();
}

void ExecutorManagerImpl::clear(const std::string& id) {
    return m_manager->clear(id);
}

std::shared_ptr<InferenceEngine::ExecutorManager> create_old_manager(
    const std::shared_ptr<ov::threading::ExecutorManager>& manager) {
    return std::make_shared<ExecutorManagerImpl>(manager);
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
            _manager = manager = create_old_manager(ov::threading::executor_manager());
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
