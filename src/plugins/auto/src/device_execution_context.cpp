// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "device_execution_context.hpp"

namespace ov {
namespace auto_plugin {

SoCompiledModel AutoCompileContext::compiled_model() const {
    const auto* device_context = m_device_context.load();
    return device_context ? device_context->compiled_model() : m_compile_result;
}

DeviceExecutionContext::DeviceExecutionContext(DeviceName name) : m_name(std::move(name)) {}

void DeviceExecutionContext::make_ready(const SoCompiledModel& model,
                                        size_t num_requests,
                                        const WorkerSetup& setup_worker) {
    OPENVINO_ASSERT(m_state != State::Ready, "Device ", m_name, " already has workers");
    m_compiled_model = model;
    try {
        m_workers.resize(num_requests);
        m_idle_workers.set_capacity(num_requests);
        int index = 0;
        for (auto& worker : m_workers) {
            worker.m_inferrequest = {model->create_infer_request(), model._so};
            worker.m_index = index++;
            OPENVINO_ASSERT(m_idle_workers.try_push(std::make_pair(worker.m_index, &worker)));
            setup_worker(*this, worker);
        }
    } catch (...) {
        // leave the context as it was, the requests go before the model
        std::pair<int, WorkerInferRequest*> idle;
        while (m_idle_workers.try_pop(idle)) {
        }
        m_idle_workers.set_capacity(0);
        m_workers.clear();
        m_compiled_model = {};
        throw;
    }
    m_state = State::Ready;
    m_last_error.clear();
}

bool DeviceExecutionContext::try_release() {
    if (m_keep_alive || m_state != State::Ready) {
        return false;
    }
    // an empty idle queue means a worker is still running
    std::vector<std::pair<int, WorkerInferRequest*>> idle_workers;
    std::pair<int, WorkerInferRequest*> idle;
    while (m_idle_workers.try_pop(idle)) {
        idle_workers.push_back(idle);
    }
    if (idle_workers.size() != m_workers.size()) {
        for (auto& worker : idle_workers) {
            m_idle_workers.try_push(std::move(worker));
        }
        return false;
    }
    collect_worker_stats();
    m_idle_workers.set_capacity(0);
    // the requests keep the compiled model alive, so they go first
    m_workers.clear();
    m_compiled_model = {};
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
    m_idle_workers.set_capacity(0);
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

const DeviceName& DeviceExecutionContext::name() const {
    return m_name;
}

const SoCompiledModel& DeviceExecutionContext::compiled_model() const {
    return m_compiled_model;
}

std::vector<WorkerInferRequest>& DeviceExecutionContext::workers() {
    return m_workers;
}

NotBusyPriorityWorkerRequests& DeviceExecutionContext::idle_workers() {
    return m_idle_workers;
}

TaskQueue& DeviceExecutionContext::pending_tasks() {
    return m_pending_tasks;
}

DeviceCacheInfo& DeviceExecutionContext::cache_info() {
    return m_cache_info;
}

void DeviceExecutionContext::collect_worker_stats() {
    for (auto& worker : m_workers) {
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
    for (const auto& request : m_workers) {
        req_all_start_times.insert(req_all_start_times.end(),
                                   request.m_start_times.begin(),
                                   request.m_start_times.end());
        req_all_end_times.insert(req_all_end_times.end(), request.m_end_times.begin(), request.m_end_times.end());
    }
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
        LOG_INFO_TAG("%s: first inference time:%lf ms", m_name.c_str(), first_infer_duration.count());
        LOG_INFO_TAG("%s:infer:%lu", m_name.c_str(), static_cast<unsigned long>(count));
        std::chrono::duration<double, std::milli> durtation =
            busy_duration.count() > 0.0 ? busy_duration : req_all_end_times.back() - time;
        LOG_INFO_TAG("%s:fps:%lf", m_name.c_str(), n * 1000 / durtation.count());
    }
}

}  // namespace auto_plugin
}  // namespace ov
