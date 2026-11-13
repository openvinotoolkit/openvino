// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cache_manager.hpp"
#include "common.hpp"

namespace ov {
namespace auto_plugin {

/**
 * @brief Everything AUTO owns for one device: compiled model, worker requests, idle/pending queues and statistics.
 *
 * State transitions happen only inside the member functions:
 *   Unloaded/Released/Faulted --make_ready()--> Ready --try_release()--> Released
 *   Unloaded/Released/Faulted --mark_faulted()--> Faulted
 *
 * Thread safety: the object takes no lock. The queues are thread safe; make_ready(), try_release() and mark_faulted()
 * must be serialized by the caller and never overlap with the completion callback of a worker that is still busy.
 */
class DeviceExecutionContext {
public:
    enum class State { Unloaded, Ready, Released, Faulted };
    // binds the completion callback of one worker, it must capture the context rather than a queue pointer
    using WorkerSetup = std::function<void(DeviceExecutionContext&, WorkerInferRequest&)>;

    explicit DeviceExecutionContext(DeviceName name);
    DeviceExecutionContext(const DeviceExecutionContext&) = delete;
    DeviceExecutionContext& operator=(const DeviceExecutionContext&) = delete;

    // Unloaded/Released/Faulted -> Ready, clears the last error; asserts if the context is already Ready
    void make_ready(const SoCompiledModel& model, size_t num_requests, const WorkerSetup& setup_worker);
    // Ready -> Released; returns false without any side effect if kept alive or any worker is busy
    bool try_release();
    // Unloaded/Released/Faulted -> Faulted, records why the model failed to compile; no-op when Ready
    void mark_faulted(const std::string& reason);
    // a context kept alive is never released, e.g. it backs the inputs()/outputs() of the compiled model
    void keep_alive();
    // no worker can be taken or returned afterwards, so a late completion callback cannot re-queue a worker
    void stop_scheduling();

    State state() const;
    bool is_ready() const;
    bool is_kept_alive() const;
    const DeviceName& name() const;
    // reason of the latest compile failure, empty unless Faulted
    const std::string& last_error() const;
    const SoCompiledModel& compiled_model() const;
    std::vector<WorkerInferRequest>& workers();
    NotBusyPriorityWorkerRequests& idle_workers();
    TaskQueue& pending_tasks();
    DeviceCacheInfo& cache_info();

    /**
     * @brief Log the infer count and fps of the device, including the workers released earlier.
     * @param serialized_inference true if the inferences never overlap, so fps is based on the busy time
     */
    void log_summary(const std::string& log_tag, Time cpuhelp_release_time, bool serialized_inference) const;
    // hands the statistics gathered by try_release() over to the caller
    std::pair<std::list<Time>, std::list<Time>> take_released_stats();

private:
    void collect_worker_stats();

    const DeviceName m_name;
    bool m_keep_alive = false;
    State m_state = State::Unloaded;
    std::string m_last_error;
    DeviceCacheInfo m_cache_info;
    std::list<Time> m_released_start_times;
    std::list<Time> m_released_end_times;
    // the declaration order is the reverse of the destruction order: requests first, the model last
    SoCompiledModel m_compiled_model;
    NotBusyPriorityWorkerRequests m_idle_workers;
    TaskQueue m_pending_tasks;
    std::vector<WorkerInferRequest> m_workers;
};

}  // namespace auto_plugin
}  // namespace ov
