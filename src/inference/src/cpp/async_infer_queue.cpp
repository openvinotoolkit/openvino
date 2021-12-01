// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/async_infer_queue.hpp"

#include <condition_variable>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "ie/ie_plugin_config.hpp"

namespace ov {
namespace runtime {
class AsyncInferQueue::Impl {
public:
    Impl() = delete;

    Impl(ExecutableNetwork& net, size_t jobs) {
        // Automatically set number of jobs
        if (jobs == 0) {
            try {
                auto parameter_value = net.get_metric(METRIC_KEY(SUPPORTED_METRICS));
                auto supported_metrics = parameter_value.as<std::vector<std::string>>();
                const std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
                if (std::find(supported_metrics.begin(), supported_metrics.end(), key) != supported_metrics.end()) {
                    parameter_value = net.get_metric(key);
                    if (parameter_value.is<unsigned int>())
                        jobs = parameter_value.as<unsigned int>();
                    else
                        throw ov::Exception("Unsupported format for " + key +
                                            "! Please specify number of jobs explicitly!");
                } else {
                    ov::Exception("Can't load model: " + key +
                                  " is not supported! Please specify number of jobs explicitly!");
                }
            } catch (const std::exception& ex) {
                ov::Exception(ex.what());
            }
        }

        for (size_t handle = 0; handle < jobs; handle++) {
            m_pool.push_back(net.create_infer_request());
            m_idle_handles.push(handle);
            m_userdata.push_back(nullptr);
        }

        set_default_callback();
    }

    InferRequest& get_request(size_t i);

    size_t get_idle_id();
    size_t pop_idle_id();

    void start_async(const ov::Any userdata);
    void start_async(std::map<size_t, ov::runtime::Tensor>& inputs, const ov::Any userdata);
    void start_async(std::map<std::string, ov::runtime::Tensor>& inputs, const ov::Any userdata);

    bool is_ready();
    void wait_all();

    void set_callback(std::function<void(std::exception_ptr, ov::runtime::InferRequest&, const ov::Any&)> callback);

private:
    void set_default_callback();

    std::vector<ov::runtime::InferRequest> m_pool;  // pool of jobs (InferRequests)
    std::queue<size_t> m_idle_handles;              // idle handles of requests
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::vector<ov::Any> m_userdata;
};

InferRequest& AsyncInferQueue::Impl::get_request(size_t i) {
    return m_pool[i];
}

size_t AsyncInferQueue::Impl::get_idle_id() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [this] {
        return !m_idle_handles.empty();
    });

    size_t request_id = m_idle_handles.front();

    return request_id;
}

size_t AsyncInferQueue::Impl::pop_idle_id() {
    // Get idle request id and pop it from queue
    size_t idle_request_id = get_idle_id();
    std::unique_lock<std::mutex> lock(m_mutex);
    m_idle_handles.pop();

    return idle_request_id;
}

void AsyncInferQueue::Impl::start_async(ov::Any userdata) {
    auto request_id = pop_idle_id();
    if (!userdata.empty()) {
        m_userdata[request_id] = userdata;
    }
    m_pool[request_id].start_async();
}

void AsyncInferQueue::Impl::start_async(std::map<size_t, ov::runtime::Tensor>& inputs, ov::Any userdata) {
    size_t request_id = pop_idle_id();
    if (!userdata.empty()) {
        m_userdata[request_id] = userdata;
    }
    for (auto const& t : inputs) {
        m_pool[request_id].set_input_tensor(t.first, t.second);
    }
    m_pool[request_id].start_async();
}

void AsyncInferQueue::Impl::start_async(std::map<std::string, ov::runtime::Tensor>& inputs, ov::Any userdata) {
    size_t request_id = pop_idle_id();
    if (!userdata.empty()) {
        m_userdata[request_id] = userdata;
    }
    for (auto const& t : inputs) {
        m_pool[request_id].set_tensor(t.first, t.second);
    }
    m_pool[request_id].start_async();
}

bool AsyncInferQueue::Impl::is_ready() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [this] {
        return !m_idle_handles.empty();
    });

    return !(m_idle_handles.empty());
}

void AsyncInferQueue::Impl::wait_all() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [this] {
        return m_idle_handles.size() == m_pool.size();
    });
}

void AsyncInferQueue::Impl::set_default_callback() {
    for (size_t handle = 0; handle < m_pool.size(); handle++) {
        m_pool[handle].set_callback([this, handle /* ... */](std::exception_ptr e) {
            // Add idle handle to queue
            std::unique_lock<std::mutex> lock(m_mutex);
            m_idle_handles.push(handle);
            // Notify locks
            m_cv.notify_one();
        });
    }
}

void AsyncInferQueue::Impl::set_callback(
    std::function<void(std::exception_ptr, ov::runtime::InferRequest&, const ov::Any&)> callback) {
    for (size_t handle = 0; handle < m_pool.size(); handle++) {
        m_pool[handle].set_callback([this, callback, handle /* ... */](std::exception_ptr e) {
            callback(e, m_pool[handle], m_userdata[handle]);
            // Add idle handle to queue
            std::unique_lock<std::mutex> lock(m_mutex);
            m_idle_handles.push(handle);
            // Notify locks
            m_cv.notify_one();
        });
    }
}

AsyncInferQueue::AsyncInferQueue(ExecutableNetwork& net, size_t jobs)
    : m_pimpl{new Impl{net, jobs}, [](Impl* impl) {
                  delete impl;
              }} {}

size_t AsyncInferQueue::get_idle_id() {
    return m_pimpl->get_idle_id();
}

void AsyncInferQueue::start_async(const ov::Any userdata) {
    m_pimpl->start_async(userdata);
}

void AsyncInferQueue::start_async(std::map<size_t, ov::runtime::Tensor> inputs, const ov::Any userdata) {
    m_pimpl->start_async(inputs, userdata);
}

void AsyncInferQueue::start_async(std::map<std::string, ov::runtime::Tensor> inputs, const ov::Any userdata) {
    m_pimpl->start_async(inputs, userdata);
}

bool AsyncInferQueue::is_ready() {
    return m_pimpl->is_ready();
}

void AsyncInferQueue::wait_all() {
    m_pimpl->wait_all();
}

void AsyncInferQueue::set_callback(
    std::function<void(std::exception_ptr, ov::runtime::InferRequest&, const ov::Any&)> callback) {
    m_pimpl->set_callback(callback);
}

InferRequest& AsyncInferQueue::operator[](size_t i) {
    return m_pimpl->get_request(i);
}

}  // namespace runtime
}  // namespace ov