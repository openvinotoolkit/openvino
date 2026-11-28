// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <deque>
#include <functional>
#include <list>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common.hpp"

namespace ov {
namespace auto_plugin {

class WorkerPool;

/**
 * @brief Exclusive use of one worker of a WorkerPool.
 *
 * The destructor gives the worker back unless the lease was detached. A lease is detached once the inference has been
 * started on the worker, the completion callback then adopts the worker into a new lease and gives it back.
 */
class WorkerLease {
public:
    WorkerLease() = default;
    WorkerLease(WorkerLease&& other) noexcept;
    WorkerLease& operator=(WorkerLease&& other) noexcept;
    WorkerLease(const WorkerLease&) = delete;
    WorkerLease& operator=(const WorkerLease&) = delete;
    ~WorkerLease();

    explicit operator bool() const {
        return m_worker != nullptr;
    }
    WorkerInferRequest* get() const {
        return m_worker;
    }
    // the in-flight inference keeps the worker, the lease forgets it
    WorkerInferRequest* detach();
    // returns false if the pool takes no worker back anymore, the lease is empty afterwards in both cases
    bool give_back();

private:
    friend class WorkerPool;
    WorkerLease(WorkerPool* pool, WorkerInferRequest* worker);

    WorkerPool* m_pool = nullptr;
    WorkerInferRequest* m_worker = nullptr;
};

/**
 * @brief Worker requests of one compiled model and the tasks that wait for one of them.
 *
 * A worker is either idle or leased, and the pool counts the leased ones under its own lock, so it knows when every
 * worker is idle without draining the idle queue. Once closed, nothing can be leased or given back.
 */
class WorkerPool {
public:
    using WorkerSetup = std::function<void(WorkerInferRequest&)>;

    WorkerPool() = default;
    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    // creates the workers on the model and opens the pool; the pool stays closed and empty if a step throws
    void open(const SoCompiledModel& model, size_t num_requests, const WorkerSetup& setup_worker);
    // closes the pool if no worker is leased and no task is queued; the workers are kept until clear()
    bool try_close_idle();
    // closes the pool unconditionally, a worker that comes back later is not taken anymore
    void close();
    // drops the workers of a closed pool, so the requests go before the model they were created on
    void clear();

    // the idle worker with the lowest index, or an empty lease
    WorkerLease try_acquire();
    // takes the worker of a detached lease back into a lease
    WorkerLease adopt(WorkerInferRequest* worker);

    void enqueue(ov::threading::Task task);
    bool try_dequeue(ov::threading::Task& task);

    size_t size() const;
    size_t idle_count() const;
    size_t leased_count() const;
    bool has_queued_tasks() const;
    // The three below read the workers without the lock: only the serialized owner that calls open(), close() and
    // clear() may use them, never concurrently with those.
    // only for binding a request to the worker with the given index
    WorkerInferRequest& worker(size_t index);
    void for_each_worker(const std::function<void(const WorkerInferRequest&)>& visit) const;
    // size(), worker(), for_each_worker() and wait_all() read the workers without a lock: open() and clear() must be
    // serialized with them by the owner, which is the case since a pool is only reopened after try_close_idle()
    // waits for the inference of every worker, a failed one is reported to on_error
    void wait_all(const std::function<void(const ov::Exception&)>& on_error);

private:
    friend class WorkerLease;
    bool give_back(WorkerInferRequest* worker);

    mutable std::mutex m_mutex;
    bool m_open = false;
    size_t m_leased = 0;
    std::set<int> m_idle;
    std::deque<ov::threading::Task> m_tasks;
    // the last member, so the requests are destroyed first
    std::vector<WorkerInferRequest> m_workers;
};

/**
 * @brief Everything AUTO owns for one model instance: the compiled model, its worker pool and the statistics.
 *
 * State transitions happen only inside the member functions:
 *   Unloaded/Released/Faulted --make_ready()--> Ready --try_release()--> Released
 *   Unloaded/Released/Faulted --mark_faulted()--> Faulted
 *
 * Thread safety: the pool, the model handle (m_model_mutex) and the state are safe to use from any thread.
 * make_ready(), try_release() and mark_faulted() must be serialized by the caller; try_release() refuses while a
 * worker is leased, so it never overlaps with the completion callback of a running inference.
 */
class DeviceExecutionContext {
public:
    enum class State { Unloaded, Ready, Released, Faulted };
    // binds the completion callback of one worker, it must capture the context rather than the pool
    using WorkerSetup = std::function<void(DeviceExecutionContext&, WorkerInferRequest&)>;

    explicit DeviceExecutionContext(ModelInstanceKey key);
    DeviceExecutionContext(const DeviceExecutionContext&) = delete;
    DeviceExecutionContext& operator=(const DeviceExecutionContext&) = delete;

    // Unloaded/Released/Faulted -> Ready, clears the last error; asserts if the context is already Ready
    void make_ready(const SoCompiledModel& model, size_t num_requests, const WorkerSetup& setup_worker);
    // Ready -> Released; returns false without any side effect if kept alive, any worker is leased or a task is
    // queued. The workers and the model are destroyed here, so do not call it while holding a lock the model queries
    // need.
    bool try_release();
    // Unloaded/Released/Faulted -> Faulted, records why the model failed to compile; no-op when Ready
    void mark_faulted(const std::string& reason);
    // a context kept alive is never released, e.g. the passthrough model that serves every request directly
    void keep_alive();
    // no worker can be leased or given back afterwards, so a late completion callback cannot re-queue a worker
    void stop_scheduling();

    State state() const;
    bool is_ready() const;
    bool is_kept_alive() const;
    const ModelInstanceKey& key() const;
    // the device the instance runs on
    const DeviceName& name() const;
    // when an inference was last dispatched to the instance; only the dynamic dispatch thread uses them
    void mark_used(Time now);
    Time last_used() const;
    // reason of the latest compile failure, empty unless Faulted
    const std::string& last_error() const;
    // a copy, so it stays valid when another thread releases the context meanwhile; empty unless Ready
    SoCompiledModel compiled_model() const;
    WorkerPool& pool();

    /**
     * @brief Log the infer count and fps of the device, including the workers released earlier.
     * @param serialized_inference true if the inferences never overlap, so fps is based on the busy time
     */
    void log_summary(const std::string& log_tag, Time cpuhelp_release_time, bool serialized_inference) const;
    // hands the statistics gathered by try_release() over to the caller
    std::pair<std::list<Time>, std::list<Time>> take_released_stats();

private:
    void collect_worker_stats();
    // swaps the model handle and returns the previous one, so that it is destroyed outside of the lock
    SoCompiledModel exchange_model(SoCompiledModel model);

    const ModelInstanceKey m_key;
    bool m_keep_alive = false;
    // written only by the serialized owner, atomic so that other threads can read it without a lock
    std::atomic<State> m_state{State::Unloaded};
    std::string m_last_error;
    Time m_last_used{};
    std::list<Time> m_released_start_times;
    std::list<Time> m_released_end_times;
    // the declaration order is the reverse of the destruction order: requests first, the model last
    // the model handle is the one member that other threads read while the context is released, hence the mutex
    mutable std::mutex m_model_mutex;
    SoCompiledModel m_compiled_model;
    WorkerPool m_pool;
};

}  // namespace auto_plugin
}  // namespace ov
