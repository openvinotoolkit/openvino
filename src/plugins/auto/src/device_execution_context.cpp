// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "device_execution_context.hpp"

namespace ov {
namespace auto_plugin {

bool RoleSlot::is_ready() const {
    const auto* device_context = m_device_context.load();
    return device_context && device_context->is_ready();
}

SoCompiledModel RoleSlot::compiled_model() const {
    const auto* device_context = m_device_context.load();
    return device_context ? device_context->compiled_model() : SoCompiledModel{};
}

WorkerLease::WorkerLease(WorkerPool* pool, WorkerInferRequest* worker) : m_pool(pool), m_worker(worker) {}

WorkerLease::WorkerLease(WorkerLease&& other) noexcept
    : m_pool(std::exchange(other.m_pool, nullptr)),
      m_worker(std::exchange(other.m_worker, nullptr)) {}

WorkerLease& WorkerLease::operator=(WorkerLease&& other) noexcept {
    if (this != &other) {
        give_back();
        m_pool = std::exchange(other.m_pool, nullptr);
        m_worker = std::exchange(other.m_worker, nullptr);
    }
    return *this;
}

WorkerLease::~WorkerLease() {
    give_back();
}

WorkerInferRequest* WorkerLease::detach() {
    m_pool = nullptr;
    return std::exchange(m_worker, nullptr);
}

bool WorkerLease::give_back() {
    auto* worker = std::exchange(m_worker, nullptr);
    auto* pool = std::exchange(m_pool, nullptr);
    return worker && pool && pool->give_back(worker);
}

void WorkerPool::open(const SoCompiledModel& model, size_t num_requests, const WorkerSetup& setup_worker) {
    std::vector<WorkerInferRequest> workers(num_requests);
    int index = 0;
    for (auto& worker : workers) {
        worker.m_inferrequest = {model->create_infer_request(), model._so};
        worker.m_index = index++;
    }
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        OPENVINO_ASSERT(!m_open && m_workers.empty(), "The worker pool is already open");
        m_workers = std::move(workers);
    }
    // the callbacks capture the address of the worker, so they are bound once the workers are in place
    try {
        for (auto& worker : m_workers) {
            setup_worker(worker);
        }
    } catch (...) {
        std::vector<WorkerInferRequest> dropped;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            dropped.swap(m_workers);
        }
        throw;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const auto& worker : m_workers) {
        m_idle.insert(worker.m_index);
    }
    m_leased = 0;
    m_open = true;
}

bool WorkerPool::try_close_idle() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_leased != 0 || !m_tasks.empty()) {
        return false;
    }
    m_open = false;
    m_idle.clear();
    return true;
}

void WorkerPool::close() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_open = false;
    m_idle.clear();
}

void WorkerPool::clear() {
    std::vector<WorkerInferRequest> dropped;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        OPENVINO_ASSERT(!m_open, "An open worker pool cannot be cleared");
        dropped.swap(m_workers);
        m_leased = 0;
    }
}

WorkerLease WorkerPool::try_acquire() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_open || m_idle.empty()) {
        return {};
    }
    const auto index = *m_idle.begin();
    m_idle.erase(m_idle.begin());
    m_leased++;
    return {this, &m_workers[index]};
}

WorkerLease WorkerPool::adopt(WorkerInferRequest* worker) {
    return {this, worker};
}

bool WorkerPool::give_back(WorkerInferRequest* worker) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_leased > 0) {
        m_leased--;
    }
    if (!m_open) {
        return false;
    }
    m_idle.insert(worker->m_index);
    return true;
}

void WorkerPool::enqueue(ov::threading::Task task) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_tasks.push_back(std::move(task));
}

bool WorkerPool::try_dequeue(ov::threading::Task& task) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_tasks.empty()) {
        return false;
    }
    task = std::move(m_tasks.front());
    m_tasks.pop_front();
    return true;
}

size_t WorkerPool::size() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_workers.size();
}

size_t WorkerPool::idle_count() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_idle.size();
}

size_t WorkerPool::leased_count() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_leased;
}

bool WorkerPool::has_queued_tasks() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return !m_tasks.empty();
}

WorkerInferRequest& WorkerPool::worker(size_t index) {
    return m_workers.at(index);
}

void WorkerPool::for_each_worker(const std::function<void(const WorkerInferRequest&)>& visit) const {
    for (const auto& worker : m_workers) {
        visit(worker);
    }
}

void WorkerPool::wait_all(const std::function<void(const ov::Exception&)>& on_error) {
    for (auto& worker : m_workers) {
        try {
            worker.m_inferrequest._ptr->wait();
        } catch (const ov::Exception& ex) {
            on_error(ex);
        }
    }
}

DeviceExecutionContext::DeviceExecutionContext(ModelInstanceKey key) : m_key(std::move(key)) {}

void DeviceExecutionContext::make_ready(const SoCompiledModel& model,
                                        size_t num_requests,
                                        const WorkerSetup& setup_worker) {
    OPENVINO_ASSERT(m_state != State::Ready, "Device ", m_key.to_string(), " already has workers");
    exchange_model(model);
    try {
        m_pool.open(model, num_requests, [this, &setup_worker](WorkerInferRequest& worker) {
            setup_worker(*this, worker);
        });
    } catch (...) {
        // leave the context as it was
        exchange_model({});
        throw;
    }
    m_state = State::Ready;
    m_last_error.clear();
}

bool DeviceExecutionContext::try_release() {
    if (m_keep_alive || m_state != State::Ready || !m_pool.try_close_idle()) {
        return false;
    }
    collect_worker_stats();
    // the requests keep the compiled model alive, so they go first
    m_pool.clear();
    exchange_model({});
    m_state = State::Released;
    return true;
}

void DeviceExecutionContext::mark_faulted(const std::string& reason) {
    if (m_state != State::Ready) {
        m_state = State::Faulted;
        m_last_error = reason;
    }
}

void DeviceExecutionContext::keep_alive() {
    m_keep_alive = true;
}

void DeviceExecutionContext::stop_scheduling() {
    m_pool.close();
}

DeviceExecutionContext::State DeviceExecutionContext::state() const {
    return m_state;
}

bool DeviceExecutionContext::is_ready() const {
    return m_state == State::Ready;
}

bool DeviceExecutionContext::is_kept_alive() const {
    return m_keep_alive;
}

const std::string& DeviceExecutionContext::last_error() const {
    return m_last_error;
}

const ModelInstanceKey& DeviceExecutionContext::key() const {
    return m_key;
}

const DeviceName& DeviceExecutionContext::name() const {
    return m_key.device;
}

SoCompiledModel DeviceExecutionContext::compiled_model() const {
    std::lock_guard<std::mutex> lock(m_model_mutex);
    return m_compiled_model;
}

SoCompiledModel DeviceExecutionContext::exchange_model(SoCompiledModel model) {
    std::lock_guard<std::mutex> lock(m_model_mutex);
    std::swap(m_compiled_model, model);
    return model;
}

void DeviceExecutionContext::mark_used(Time now) {
    m_last_used = now;
}

Time DeviceExecutionContext::last_used() const {
    return m_last_used;
}

WorkerPool& DeviceExecutionContext::pool() {
    return m_pool;
}

void DeviceExecutionContext::collect_worker_stats() {
    for (size_t i = 0; i < m_pool.size(); i++) {
        auto& worker = m_pool.worker(i);
        m_released_start_times.splice(m_released_start_times.end(), worker.m_start_times);
        m_released_end_times.splice(m_released_end_times.end(), worker.m_end_times);
    }
}

std::pair<std::list<Time>, std::list<Time>> DeviceExecutionContext::take_released_stats() {
    return {std::move(m_released_start_times), std::move(m_released_end_times)};
}

void DeviceExecutionContext::log_summary(const std::string& log_tag,
                                         Time cpuhelp_release_time,
                                         bool serialized_inference) const {
    const auto get_log_tag = [&log_tag]() -> const std::string& {
        return log_tag;
    };
    std::list<Time> req_all_start_times = m_released_start_times;
    std::list<Time> req_all_end_times = m_released_end_times;
    m_pool.for_each_worker([&](const WorkerInferRequest& request) {
        req_all_start_times.insert(req_all_start_times.end(),
                                   request.m_start_times.begin(),
                                   request.m_start_times.end());
        req_all_end_times.insert(req_all_end_times.end(), request.m_end_times.begin(), request.m_end_times.end());
    });
    size_t count = req_all_start_times.size();
    OPENVINO_ASSERT(count == req_all_end_times.size());
    std::chrono::duration<double, std::milli> first_infer_duration =
        (count != 0) ? req_all_end_times.front() - req_all_start_times.front()
                     : std::chrono::duration<double, std::milli>(0.0);
    req_all_start_times.sort(std::less<Time>());
    req_all_end_times.sort(std::less<Time>());
    // the inferences are serialized there, so the sum is the time the device was busy
    std::chrono::duration<double, std::milli> busy_duration{0.0};
    if (serialized_inference) {
        for (auto start = req_all_start_times.begin(), end = req_all_end_times.begin();
             start != req_all_start_times.end();
             ++start, ++end) {
            busy_duration += *end - *start;
        }
    }
    auto n = req_all_start_times.size();
    Time time;
    while (!req_all_start_times.empty()) {
        time = req_all_start_times.front();
        if (time < cpuhelp_release_time) {
            req_all_start_times.pop_front();
            n--;
        } else {
            break;
        }
    }
    if (n >= 1) {
        const auto name = m_key.to_string();
        LOG_INFO_TAG("%s: first inference time:%lf ms", name.c_str(), first_infer_duration.count());
        LOG_INFO_TAG("%s:infer:%lu", name.c_str(), static_cast<unsigned long>(count));
        std::chrono::duration<double, std::milli> durtation =
            busy_duration.count() > 0.0 ? busy_duration : req_all_end_times.back() - time;
        LOG_INFO_TAG("%s:fps:%lf", name.c_str(), n * 1000 / durtation.count());
    }
}

}  // namespace auto_plugin
}  // namespace ov
